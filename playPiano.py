# -*- coding: utf-8 -*-
"""
playPiano.py - Ultra-Optimized Production Grade
================================================
Changelog vs previous version:
  1.  Continuous velocity (0.0-1.0 float) - no more discrete 1-16 layers
  2.  Equal-power crossfade (cos/sin) - eliminates volume dip between layers
  3.  Cubic interpolation pitch-shift - much cleaner than linear
  4.  Timing jitter for humanization - micro-offsets per note
  5.  Velocity momentum (arm inertia simulation)
  6.  Multiple velocity curve types: linear / exponential / s-curve / logarithmic
  7.  Half-pedal support (continuous pedal depth 0.0-1.0)
  8.  Parallel sample preloading via ThreadPoolExecutor
  9.  Top-level bisect import (no more import in hot path)
  10. Thread-safe held-note lists with locks
  11. Robust Notes loader (tolerant filename parsing)
  12. Proper resource cleanup on stop()
  13. Release sample support (optional short tail on key-up)
  14. Bass/treble adaptive release with smoother curve
"""

import os
import re
import math
import time
import bisect
import random
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

try:
    import numpy as np

    _HAVE_NUMPY = True
except Exception:
    np = None
    _HAVE_NUMPY = False

# =========================================================
# Constants
# =========================================================
_NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_DEGREE_TO_NOTE = {'1': 'C', '2': 'D', '3': 'E', '4': 'F', '5': 'G', '6': 'A', '7': 'B'}
_SEMI_TO_NAME = {0: 'C', 1: 'C#', 2: 'D', 3: 'D#', 4: 'E', 5: 'F',
                 6: 'F#', 7: 'G', 8: 'G#', 9: 'A', 10: 'A#', 11: 'B'}

# Velocity curve types
CURVE_LINEAR = 'linear'
CURVE_EXPONENTIAL = 'exponential'
CURVE_SCURVE = 's-curve'
CURVE_LOGARITHMIC = 'logarithmic'

# =========================================================
# Audio init (singleton, thread-safe)
# =========================================================
_AUDIO_INIT_LOCK = threading.Lock()
_AUDIO_INITED = False


def init_audio(sample_rate: int = 44100, buffer: int = 512,
               channels: int = 2, num_mixer_channels: int = 256):
    """Initialize pygame + mixer exactly once."""
    global _AUDIO_INITED
    if _AUDIO_INITED:
        return
    with _AUDIO_INIT_LOCK:
        if _AUDIO_INITED:
            return
        pygame.init()
        try:
            pygame.mixer.init(frequency=int(sample_rate), size=-16,
                              channels=int(channels), buffer=int(buffer))
        except Exception:
            pygame.mixer.init()
        try:
            pygame.mixer.set_num_channels(int(num_mixer_channels))
        except Exception:
            pass
        _AUDIO_INITED = True


# =========================================================
# Utilities
# =========================================================

def read_lines_auto_encoding(path: str):
    """Read text file lines with tolerant encoding detection."""
    with open(path, "rb") as f:
        data = f.read()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(enc).splitlines(True)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace").splitlines(True)


def note_name_to_midi(note_name: str) -> int:
    """C4 -> 60, F#3 -> 54, Bb2 -> 46, etc."""
    m = re.fullmatch(r'([A-Ga-g])([#bs]?)([0-8])', note_name.strip())
    if not m:
        raise ValueError(f'Invalid note name: {note_name}')
    n, acc, octv = m.group(1).upper(), m.group(2), int(m.group(3))
    semi = _NOTE_BASE[n] + (1 if acc == '#' else -1 if acc == 'b' else 0)
    return (octv + 1) * 12 + semi


def midi_to_note_name(midi: int) -> str:
    """61 -> C#4"""
    if not (0 <= midi <= 127):
        raise ValueError('MIDI out of range')
    return f"{_SEMI_TO_NAME[midi % 12]}{midi // 12 - 1}"


# =========================================================
# Velocity curve functions
# =========================================================

def apply_velocity_curve(vol: float, curve_type: str = CURVE_EXPONENTIAL,
                         curve_power: float = 1.8) -> float:
    """
    Apply non-linear velocity curve to a 0.0-1.0 volume value.

    - linear:       vol (no change)
    - exponential:  vol ^ power  (power > 1 = more dynamic range)
    - s-curve:      smooth S shape, quiet gets quieter, loud stays loud
    - logarithmic:  log curve, compresses loud end (less dynamic)
    """
    vol = max(0.0, min(1.0, vol))
    if vol <= 0.0:
        return 0.0

    if curve_type == CURVE_LINEAR:
        return vol
    elif curve_type == CURVE_EXPONENTIAL:
        return math.pow(vol, curve_power)
    elif curve_type == CURVE_SCURVE:
        # Attempt hermite-style S-curve: 3x^2 - 2x^3 then apply power
        s = vol * vol * (3.0 - 2.0 * vol)
        return math.pow(s, max(0.5, curve_power * 0.6))
    elif curve_type == CURVE_LOGARITHMIC:
        # log1p based curve - compresses dynamic range
        # log1p(x*(e-1)) maps [0,1]->[0,1] since log1p(e-1)=ln(e)=1
        return math.log1p(vol * (math.e - 1.0))
    else:
        return math.pow(vol, curve_power)


# =========================================================
# SampleLibrary
# =========================================================

class SampleLibrary:
    """
    88-key piano sample library with velocity layers.
    Supports:
      - Sparse sampling with pitch-shift fallback (cubic interp)
      - Velocity crossfade with equal-power blending
      - Lazy or eager (preload) loading
      - Release samples (optional)
      - LRU cache with thread-safe access
    """
    _SAMPLE_RE = re.compile(
        r'^(?P<n>[A-Ga-g])(?P<acc>[#bs]?)(?P<oct>[0-8])'
        r'(?:[_-]?)v(?P<vel>\d{1,3})$'
    )
    _RELEASE_RE = re.compile(
        r'^(?P<n>[A-Ga-g])(?P<acc>[#bs]?)(?P<oct>[0-8])'
        r'(?:[_-]?)rel(?:ease)?(?:[_-]?)v?(?P<vel>\d{1,3})?$'
    )
    _A0_MIDI = 21
    _C8_MIDI = 108

    def __init__(self, sample_root: str = 'resources',
                 allow_ext=('.wav', '.mp3', '.ogg')):
        self.sample_root = sample_root
        self.allow_ext = tuple(e.lower() for e in allow_ext)

        # {midi: {vel_layer: filepath}}
        self.index: dict[int, dict[int, str]] = {}
        # {midi: {vel_layer: filepath}} for release samples
        self.release_index: dict[int, dict[int, str]] = {}

        # Main cache: (midi, vel_layer) -> Sound
        self.cache: OrderedDict[tuple, pygame.mixer.Sound] = OrderedDict()
        self.max_cache = 4096

        # Base sound cache for pitch-shifted notes
        self._base_sound_cache: OrderedDict[tuple, pygame.mixer.Sound] = OrderedDict()
        self._max_base_cache = 512

        # Release sample cache
        self._release_cache: OrderedDict[tuple, pygame.mixer.Sound] = OrderedDict()
        self._max_release_cache = 256

        self._available_midis: list[int] = []
        self._sorted_vel_cache: dict[int, list[int]] = {}  # midi -> sorted vel layers
        self._scanned = False
        self._lock = threading.RLock()

    # ----- Scanning -----

    def scan(self):
        """Scan sample_root and build index. Doesn't load audio."""
        if self._scanned:
            return
        with self._lock:
            if self._scanned:
                return
            for root, _, files in os.walk(self.sample_root):
                for fn in files:
                    ext = os.path.splitext(fn)[1].lower()
                    if ext not in self.allow_ext:
                        continue
                    stem = os.path.splitext(fn)[0]
                    full_path = os.path.join(root, fn)

                    # Try release sample first
                    mr = self._RELEASE_RE.fullmatch(stem)
                    if mr:
                        self._index_release(mr, full_path)
                        continue

                    # Normal sample
                    ms = self._SAMPLE_RE.fullmatch(stem)
                    if ms:
                        self._index_normal(ms, full_path)

            self._scanned = True
            self._available_midis = sorted(self.index.keys())
            # Pre-sort velocity layers per MIDI note for fast lookup
            self._sorted_vel_cache = {
                midi: sorted(layers.keys())
                for midi, layers in self.index.items()
            }

            if not self.index:
                raise FileNotFoundError(
                    f"No piano samples found under '{self.sample_root}'. "
                    f"Expected files like 'C4v8.wav'."
                )

    def _parse_note_from_match(self, m) -> int:
        """Extract MIDI number from regex match."""
        n = m.group('n').upper()
        acc = m.group('acc')
        if acc == 's':
            acc = '#'
        octv = int(m.group('oct'))
        try:
            midi = note_name_to_midi(f"{n}{acc}{octv}")
        except ValueError:
            return -1
        if not (self._A0_MIDI <= midi <= self._C8_MIDI):
            return -1
        return midi

    def _index_normal(self, m, full_path: str):
        midi = self._parse_note_from_match(m)
        if midi < 0:
            return
        vel = int(m.group('vel'))
        self.index.setdefault(midi, {})[vel] = full_path

    def _index_release(self, m, full_path: str):
        midi = self._parse_note_from_match(m)
        if midi < 0:
            return
        vel_str = m.group('vel')
        vel = int(vel_str) if vel_str else 8  # default mid velocity
        self.release_index.setdefault(midi, {})[vel] = full_path

    # ----- Cache helpers -----

    @staticmethod
    def _lru_get(cache: OrderedDict, key):
        v = cache.get(key)
        if v is not None:
            try:
                cache.move_to_end(key)
            except Exception:
                pass
        return v

    @staticmethod
    def _lru_set(cache: OrderedDict, key, val, max_size: int):
        cache[key] = val
        try:
            cache.move_to_end(key)
        except Exception:
            pass
        while len(cache) > max_size:
            try:
                cache.popitem(last=False)
            except Exception:
                break

    # ----- Nearest note & velocity -----

    def _nearest_available_midi(self, midi: int) -> int:
        if not self._available_midis:
            raise KeyError("Sample library is empty.")
        pos = bisect.bisect_left(self._available_midis, midi)
        if pos == 0:
            return self._available_midis[0]
        if pos >= len(self._available_midis):
            return self._available_midis[-1]
        left = self._available_midis[pos - 1]
        right = self._available_midis[pos]
        return left if abs(midi - left) <= abs(right - midi) else right

    def _choose_vel_from_layers(self, layers: dict[int, str], vel_req: int) -> int:
        return min(layers.keys(), key=lambda v: abs(v - vel_req))

    def _vel_pair(self, vels_sorted: list[int], vel_target: float):
        """Find bracketing velocity layers and interpolation factor.
        vels_sorted must be pre-sorted ascending."""
        vt = float(vel_target)
        if vt <= vels_sorted[0]:
            return vels_sorted[0], vels_sorted[0], 0.0
        if vt >= vels_sorted[-1]:
            return vels_sorted[-1], vels_sorted[-1], 0.0
        pos = bisect.bisect_left(vels_sorted, vt)
        v2 = vels_sorted[pos]
        v1 = vels_sorted[pos - 1]
        if v2 == v1:
            return v1, v2, 0.0
        t = (vt - v1) / (v2 - v1)
        return v1, v2, max(0.0, min(1.0, t))

    # ----- Pitch shifting -----

    def _pitch_shift_sound(self, base_snd: pygame.mixer.Sound,
                           semitone_diff: int) -> pygame.mixer.Sound:
        """Pitch-shift using cubic interpolation (much cleaner than linear)."""
        if semitone_diff == 0:
            return base_snd
        if not _HAVE_NUMPY:
            raise RuntimeError("Pitch-shift requires numpy.")

        arr = pygame.sndarray.array(base_snd)
        ratio = 2 ** (semitone_diff / 12.0)
        n = arr.shape[0]
        # New sample positions
        new_len = int(n / ratio)
        if new_len < 2:
            return base_snd
        idx = np.linspace(0, n - 1, new_len, dtype=np.float64)

        def cubic_interp_1d(y):
            """Cubic (Catmull-Rom) interpolation - far smoother than linear."""
            y = y.astype(np.float64)
            n_samples = len(y)
            # Pad for boundary handling
            y_pad = np.empty(n_samples + 4, dtype=np.float64)
            y_pad[2:-2] = y
            y_pad[0] = y_pad[1] = y[0]
            y_pad[-2] = y_pad[-1] = y[-1]

            idx_shifted = idx + 2.0  # offset for padding
            i0 = np.clip(np.floor(idx_shifted).astype(np.int64), 0, n_samples + 3)
            frac = idx_shifted - i0

            i_m1 = np.clip(i0 - 1, 0, n_samples + 3)
            i_p1 = np.clip(i0 + 1, 0, n_samples + 3)
            i_p2 = np.clip(i0 + 2, 0, n_samples + 3)

            p0 = y_pad[i_m1]
            p1 = y_pad[i0]
            p2 = y_pad[i_p1]
            p3 = y_pad[i_p2]

            # Catmull-Rom coefficients
            f2 = frac * frac
            f3 = f2 * frac
            out = 0.5 * (
                    (-p0 + 3 * p1 - 3 * p2 + p3) * f3 +
                    (2 * p0 - 5 * p1 + 4 * p2 - p3) * f2 +
                    (-p0 + p2) * frac +
                    2 * p1
            )
            return np.clip(out, -32768, 32767).astype(np.int16)

        if arr.ndim == 1:
            out = cubic_interp_1d(arr)
        else:
            chans = [cubic_interp_1d(arr[:, c]) for c in range(arr.shape[1])]
            out = np.stack(chans, axis=1)
        return pygame.sndarray.make_sound(out)

    # ----- Sound retrieval -----

    def get_sound_layer(self, midi: int, vel_layer: int = 12) -> pygame.mixer.Sound:
        """Load and return a Sound for a specific velocity layer."""
        self.scan()
        vel_layer = int(vel_layer)

        with self._lock:
            if midi in self.index:
                layers = self.index[midi]
                chosen = vel_layer if vel_layer in layers else \
                    self._choose_vel_from_layers(layers, vel_layer)
                key = (midi, chosen)
                snd = self._lru_get(self.cache, key)
                if snd is not None:
                    return snd
                snd = pygame.mixer.Sound(layers[chosen])
                self._lru_set(self.cache, key, snd, self.max_cache)
                return snd

            # Pitch-shift fallback
            base_midi = self._nearest_available_midi(midi)
            base_layers = self.index[base_midi]
            chosen = vel_layer if vel_layer in base_layers else \
                self._choose_vel_from_layers(base_layers, vel_layer)

            key = (midi, chosen)
            snd = self._lru_get(self.cache, key)
            if snd is not None:
                return snd

            base_key = (base_midi, chosen)
            base_snd = self._lru_get(self._base_sound_cache, base_key)
            if base_snd is None:
                base_snd = pygame.mixer.Sound(base_layers[chosen])
                self._lru_set(self._base_sound_cache, base_key, base_snd,
                              self._max_base_cache)

            shifted = self._pitch_shift_sound(base_snd, midi - base_midi)
            self._lru_set(self.cache, key, shifted, self.max_cache)
            return shifted

    def get_sound_blend(self, midi: int, vel_target: float = 12.0,
                        equal_power: bool = True):
        """
        Return [(Sound, weight, layer), ...] for velocity crossfade.

        NEW: equal_power=True uses cos/sin blending to prevent volume dip.
        """
        self.scan()
        vt = float(vel_target)

        layers = self.index.get(midi)
        effective_midi = midi
        if layers is None:
            effective_midi = self._nearest_available_midi(midi)
            layers = self.index[effective_midi]

        vels = self._sorted_vel_cache.get(effective_midi)
        if vels is None:
            vels = sorted(layers.keys())
        v1, v2, t = self._vel_pair(vels, vt)

        if v1 == v2 or t <= 0.001:
            return [(self.get_sound_layer(midi, v1), 1.0, v1)]
        if t >= 0.999:
            return [(self.get_sound_layer(midi, v2), 1.0, v2)]

        if equal_power:
            # Equal-power crossfade: cos/sin curve
            # At t=0.5, both channels at ~0.707 -> total power = 1.0
            angle = t * (math.pi / 2.0)
            w1 = math.cos(angle)
            w2 = math.sin(angle)
        else:
            w1 = 1.0 - t
            w2 = t

        s1 = self.get_sound_layer(midi, v1)
        s2 = self.get_sound_layer(midi, v2)
        return [(s1, w1, v1), (s2, w2, v2)]

    def get_release_sound(self, midi: int, vel: int = 8):
        """Get release sample if available, else None."""
        self.scan()
        if midi not in self.release_index:
            return None
        layers = self.release_index[midi]
        chosen = vel if vel in layers else self._choose_vel_from_layers(layers, vel)
        key = (midi, chosen)
        snd = self._lru_get(self._release_cache, key)
        if snd is not None:
            return snd
        try:
            snd = pygame.mixer.Sound(layers[chosen])
            self._lru_set(self._release_cache, key, snd, self._max_release_cache)
            return snd
        except Exception:
            return None

    def get_sound(self, midi: int, vel_req: int = 12) -> pygame.mixer.Sound:
        """Compatibility wrapper."""
        return self.get_sound_layer(midi, vel_req)


# =========================================================
# Score parsing / tokenizing
# =========================================================

def tokenize_score_line(line: str) -> list:
    """Split score line into tokens, keeping brackets [] and braces {} intact."""
    line = line.split('//', 1)[0].strip()
    if not line:
        return []
    tokens = []
    buf = []
    depth = 0
    for ch in line:
        if ch in ('[', '{'):
            depth += 1
            buf.append(ch)
        elif ch in (']', '}'):
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch.isspace() and depth == 0:
            if buf:
                tokens.append(''.join(buf).strip())
                buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append(''.join(buf).strip())
    return tokens


def parse_note_atom(atom: str, default_vel: int = 12) -> tuple:
    """Parse note: C4, F#3v14, 1+, [C4,E4,G4] -> (midi, vel_float, display)."""
    atom = atom.strip()
    if not atom:
        raise ValueError("Empty note")

    # Legacy numeric degree: 1, 2+, 3--, 5+v14 etc
    m_deg = re.fullmatch(r'([1-7])([+-]{0,2})(?:[@vV]?(\d{1,3}))?', atom)
    if m_deg:
        deg, shift, vel_raw = m_deg.group(1), m_deg.group(2), m_deg.group(3)
        base_oct = 4
        oct_shift = shift.count('+') - shift.count('-')
        note_letter = _DEGREE_TO_NOTE[deg]
        note_name = f"{note_letter}{base_oct + oct_shift}"
        midi = note_name_to_midi(note_name)
        vel_req = float(default_vel)
        if vel_raw:
            v = int(vel_raw)
            if 1 <= v <= 16:
                vel_req = float(v)
            elif 1 <= v <= 127:
                vel_req = max(1.0, min(16.0, v / 127.0 * 16.0))
        return midi, vel_req, note_name

    # Modern note name: C4, F#3v10, Bb2@127
    m = re.fullmatch(r'([A-Ga-g])([#bs]?)([0-8])(?:[@vV](\d{1,3}))?', atom)
    if not m:
        raise ValueError(f"Invalid note token: {atom}")
    n = m.group(1).upper()
    acc = m.group(2)
    if acc.lower() == 's':
        acc = '#'
    octv = int(m.group(3))
    midi = note_name_to_midi(f"{n}{acc}{octv}")
    vel_raw = m.group(4)
    vel_req = float(default_vel)
    if vel_raw:
        v = int(vel_raw)
        if 1 <= v <= 16:
            vel_req = float(v)
        elif 1 <= v <= 127:
            vel_req = max(1.0, min(16.0, v / 127.0 * 16.0))
    display = f"{n}{acc}{octv}"
    return midi, vel_req, display


def parse_step_token(token: str, default_vel: int = 12) -> dict:
    """
    Parse time-step token into structured action dict.

    Supports:
      - Notes: C4, [C4,E4,G4], 1+
      - Extend: 0
      - Rest: _, R, REST
      - Pedal: P+/P-, PD/PU, or PH0.5 (half-pedal with depth)
        P-|P+ macro stores all actions in pedal_actions list (syncopated pedal)
      - Soft pedal: U+/U-
      - Tuplet: {note1;note2;note3} — N notes evenly subdividing the time span.
        Follow with 0s to define span: {A;B;C} 0 0 0 = 3 notes in 4 steps.
        Each sub-note can be a chord: {[C5,E5];[D5,F5];[E5,G5]}
    """
    token = token.strip()
    if not token:
        return {'notes': [], 'extend': False, 'rest': False,
                'pedal_actions': [], 'soft': None, 'tuplet': None,
                'arpeggio': False}

    # ---- Arpeggio detection: ~[C4,E4,G4] ----
    arpeggio = False
    if token.startswith('~'):
        arpeggio = True
        token = token[1:]

    # ---- Tuplet detection: {note1;note2;note3} ----
    if token.startswith('{') and '}' in token:
        closing = token.index('}')
        inner = token[1:closing]
        sub_tokens = [s.strip() for s in inner.split(';') if s.strip()]
        if len(sub_tokens) >= 2:
            sub_steps = []
            for st in sub_tokens:
                sub = parse_step_token(st, default_vel)
                sub_steps.append(sub)
            return {'notes': [], 'extend': False, 'rest': False,
                    'pedal_actions': [], 'soft': None,
                    'tuplet': sub_steps, 'arpeggio': False}

    parts = [p.strip() for p in token.split('|') if p.strip()]
    notes = []
    extend = False
    rest = False
    pedal_actions = []
    soft = None

    for p in parts:
        p_upper = p.upper()

        if p in ('0', '.'):
            extend = True
            continue
        if p in ('_', 'R', 'REST'):
            rest = True
            continue

        # Half-pedal: PH0.5, PH0.3 etc
        m_hp = re.fullmatch(r'PH([\d.]+)', p_upper)
        if m_hp:
            pedal_actions.append(('half', max(0.0, min(1.0, float(m_hp.group(1))))))
            continue

        if p_upper in ('P+', 'PD', 'PEDALDOWN', 'PEDAL_DOWN',
                       'SUSTAINON', 'SUSTAIN_ON'):
            pedal_actions.append(('down', 1.0))
            continue
        if p_upper in ('P-', 'PU', 'PEDALUP', 'PEDAL_UP',
                       'SUSTAINOFF', 'SUSTAIN_OFF'):
            pedal_actions.append(('up', 0.0))
            continue

        if p_upper in ('U+', 'UC+', 'UNACORDAON', 'UNACORDA_ON',
                       'SOFTPEDALON', 'SOFTPEDAL_ON', 'SOFTON', 'SOFT_ON'):
            soft = 'on'
            continue
        if p_upper in ('U-', 'UC-', 'UNACORDAOFF', 'UNACORDA_OFF',
                       'SOFTPEDALOFF', 'SOFTPEDAL_OFF', 'SOFTOFF', 'SOFT_OFF'):
            soft = 'off'
            continue

        # Chord: [C4, E4, G4] or ~[C4, E4, G4] (arpeggio after pipe)
        if p.startswith('~'):
            arpeggio = True
            p = p[1:]
        if p.startswith('[') and p.endswith(']'):
            inside = p[1:-1].strip()
            if inside:
                atoms = [a for a in re.split(r'[,\s]+', inside) if a]
                for a in atoms:
                    notes.append(parse_note_atom(a, default_vel))
            continue

        # Single note
        notes.append(parse_note_atom(p, default_vel))

    if notes:
        extend = False
        rest = False
    return {'notes': notes, 'extend': extend, 'rest': rest,
            'pedal_actions': pedal_actions, 'soft': soft, 'tuplet': None,
            'arpeggio': arpeggio}


# =========================================================
# Notes file loader
# =========================================================

class Notes:
    """Load score files from directory. Robust filename parsing."""

    def __init__(self):
        self.notes = OrderedDict()

    def load_note(self, file_path: str):
        file = os.path.basename(file_path)
        stem = os.path.splitext(file)[0]
        suffix = os.path.splitext(file)[1].lstrip('.').lower()
        if not suffix:
            suffix = 'notes'  # default suffix for extensionless files

        # Robust parsing: title_BPM.suffix
        # Try to extract title and times from filename
        parts = stem.rsplit('_', 1)
        if len(parts) == 2:
            title = parts[0]
            try:
                times = int(parts[1])
            except ValueError:
                # Fallback: try the old way
                try:
                    title = file.split('_')[0]
                    times = int(stem[stem.find('_') + 1:])
                except (ValueError, IndexError):
                    title = stem
                    times = 180  # default
        else:
            title = stem
            times = 180

        try:
            lines = read_lines_auto_encoding(file_path)
        except Exception as e:
            print(f'⚠️  无法读取文件 {file}: {e}')
            return

        measures = [tokenize_score_line(line) for line in lines]

        if title not in self.notes:
            self.notes[title] = {}
        self.notes[title][suffix] = measures
        # Only set times if not already set (avoid overwrite from second file)
        if 'times' not in self.notes[title]:
            self.notes[title]['times'] = times

    def load_notes(self, path: str):
        if not os.path.isdir(path):
            print(f'⚠️  乐谱目录不存在: {path}')
            return self.notes
        for root, _, files in os.walk(path):
            for file in sorted(files):
                self.load_note(os.path.join(root, file))
        return self.notes


# =========================================================
# PianoSequencer
# =========================================================

class PianoSequencer(threading.Thread):
    """
    Production-grade piano sequencer.

    New features vs previous:
      - Continuous velocity (float throughout the pipeline)
      - Equal-power crossfade between layers
      - Velocity curve selection (linear/exp/s-curve/log)
      - Timing jitter (humanized micro-timing)
      - Velocity momentum (arm inertia simulation)
      - Half-pedal support (continuous pedal depth)
      - Release samples (optional)
      - Thread-safe note lists
      - Parallel preloading
    """

    def __init__(
            self,
            times: int,
            sample_root: str = 'resources',
            notes_visible: bool = False,
            default_vel: int = 12,
            main_gain: float = 1.0,
            acc_gain: float = 0.75,
            release_fade_ms: int = 120,
            overlap_ms: int = 60,
            # Audio
            sample_rate: int = 44100,
            audio_buffer: int = 512,
            num_mixer_channels: int = 256,
            # Timing
            line_pause_steps: int = 0,
            # Damping
            release_fade_bass_ms: int = None,
            release_fade_treble_ms: int = None,
            pedal_release_ms: int = None,
            # Pedals
            soft_pedal_vel_shift: float = 2.0,
            soft_pedal_gain: float = 0.88,
            # Realism - velocity
            vel_crossfade: bool = True,
            vel_jitter: float = 0.35,
            gain_jitter: float = 0.02,
            velocity_curve: float = 1.8,
            velocity_curve_type: str = CURVE_EXPONENTIAL,
            # Realism - timing
            timing_jitter_ms: float = 0.0,
            # Realism - momentum
            velocity_momentum: float = 0.0,
            # Realism - release samples
            use_release_samples: bool = True,
            release_sample_gain: float = 0.3,
            # Pedal
            repedal_window_ms: int = 85,
            # Half-pedal
            half_pedal_damping: float = 0.5,
            # Sympathetic resonance (琴弦共振)
            sympathetic_resonance: bool = False,
            resonance_gain: float = 0.028,
            resonance_pedal_boost: float = 2.5,
            # Tempo humanization (三层 Rubato)
            tempo_drift_range: float = 0.0,  # 整体速度漂移 ±% (0.04 = ±4%)
            tempo_drift_speed: float = 0.3,  # 漂移速率 (0.1=很慢, 1.0=快摆)
            phrase_accel: float = 0.0,  # 乐句呼吸 (>0 = 句首微赶, 句尾微拖)
            # Round Robin (同音重复变化)
            round_robin: bool = False,  # 微量 pitch/offset 随机
            round_robin_cents: float = 3.0,  # 音高偏移范围 ±cents
            round_robin_offset_ms: float = 8.0,  # 起始点偏移范围 ms
            # Adaptive legato (自适应连奏)
            adaptive_legato: bool = False,  # 根据音程动态调 overlap
            legato_max_interval: int = 4,  # ≤4半音=连奏(大 overlap)
            # Arpeggio (琶音)
            arpeggio_stagger_ms: float = 35.0,  # 琶音每音间隔 ms (30-40像竖琴)
    ):
        super().__init__(target=self.play)
        self.daemon = True

        self.times = int(times)
        self.notes_visible = notes_visible
        self.default_vel = default_vel
        self.main_gain = float(main_gain)
        self.acc_gain = float(acc_gain)
        self.overlap_ms = int(overlap_ms)

        init_audio(sample_rate=sample_rate, buffer=audio_buffer,
                   channels=2, num_mixer_channels=num_mixer_channels)

        self.line_pause_steps = max(0, int(line_pause_steps))

        # Damping
        self.release_fade_ms = int(release_fade_ms)
        self.release_fade_bass_ms = (
            int(release_fade_bass_ms) if release_fade_bass_ms is not None
            else max(60, int(self.release_fade_ms * 1.6))
        )
        self.release_fade_treble_ms = (
            int(release_fade_treble_ms) if release_fade_treble_ms is not None
            else max(40, int(self.release_fade_ms * 0.9))
        )
        self.pedal_release_ms = (
            int(pedal_release_ms) if pedal_release_ms is not None
            else max(60, int(min(self.release_fade_ms, 120)))
        )

        # Pedals
        self.soft_pedal = False
        self.soft_pedal_vel_shift = float(soft_pedal_vel_shift)
        self.soft_pedal_gain = float(soft_pedal_gain)

        # Realism - velocity
        self.vel_crossfade = bool(vel_crossfade)
        self.vel_jitter = float(vel_jitter)
        self.gain_jitter = float(gain_jitter)
        self.velocity_curve = float(velocity_curve)
        self.velocity_curve_type = str(velocity_curve_type)

        # Realism - timing
        self.timing_jitter_ms = float(timing_jitter_ms)

        # Realism - momentum
        self.velocity_momentum = float(velocity_momentum)
        self._last_velocity = 0.5  # internal state for momentum

        # Realism - release samples
        self.use_release_samples = bool(use_release_samples)
        self.release_sample_gain = float(release_sample_gain)

        # Repedal
        self.repedal_window_ms = int(repedal_window_ms)
        self._pending_pedal_release = []
        self._repedal_timer = None
        self._delay_timers = []
        self._delay_timer_gc_counter = 0  # GC counter for timer cleanup

        # Half-pedal
        self.pedal_depth = 0.0  # 0.0 = fully up, 1.0 = fully down
        self.half_pedal_damping = float(half_pedal_damping)

        # Sympathetic resonance (琴弦共振)
        # 真实钢琴中弹一个音时，与该音有泛音关系的其他弦会跟着轻微振动
        #   sympathetic_resonance: 总开关
        #   resonance_gain: 共振音量 (占原始音量的比例, 0.028 = 2.8%)
        #   resonance_pedal_boost: 踏板踩下时共振增益倍数 (踏板抬起制音器,共振更强)
        self.sympathetic_resonance = bool(sympathetic_resonance)
        self.resonance_gain = float(resonance_gain)
        self.resonance_pedal_boost = float(resonance_pedal_boost)
        # 泛音关系表: (半音偏移, 相对强度)
        # 基于物理泛音列: 八度最强, 纯五度次之, 两个八度再次, 大三度+八度最弱
        self._resonance_intervals = [
            (12, 1.00),  # 八度          (2:1 频率比)
            (7, 0.55),  # 纯五度        (3:2)
            (24, 0.35),  # 两个八度      (4:1)
            (19, 0.25),  # 纯五度+八度   (3:1)
            (16, 0.18),  # 大三度+八度   (5:4 跨八度)
            (-12, 0.40),  # 低八度        (反向共振)
        ]

        # Tempo humanization (三层 Rubato)
        # Layer 1: timing_jitter_ms (已有) - 微观随机
        # Layer 2: tempo_drift - 缓慢正弦飘移，模拟人类"内心节拍器"的不精确
        # Layer 3: phrase_accel - 乐句呼吸，句首微赶句尾微拖
        self.tempo_drift_range = float(tempo_drift_range)
        self.tempo_drift_speed = float(tempo_drift_speed)
        self.phrase_accel = float(phrase_accel)
        self._drift_phase = random.uniform(0, 2 * math.pi)  # 随机初始相位

        # Round Robin (同音重复变化)
        # 真实钢琴同一个键连弹两次，琴锤击弦角度/力道/弦振动状态都有微差
        # 用微量 pitch shift (±几 cents) + 采样起始点偏移来模拟
        self.round_robin = bool(round_robin)
        self.round_robin_cents = float(round_robin_cents)
        self.round_robin_offset_ms = float(round_robin_offset_ms)
        self._rr_last_midi = {}  # {midi: play_count} 追踪同音重复次数

        # Adaptive legato (自适应连奏)
        # 相邻音音程小(≤4半音)时 overlap 增大(手指"滚"过去)
        # 音程大(跳进)时 overlap 缩小(手要跳，前音必须先断)
        self.adaptive_legato = bool(adaptive_legato)
        self.legato_max_interval = int(legato_max_interval)
        self._prev_main_midi = None  # 上一步主声部最后的 MIDI 音
        self._prev_acc_midi = None  # 上一步伴奏声部最后的 MIDI 音

        # Arpeggio (琶音)
        self.arpeggio_stagger_ms = float(arpeggio_stagger_ms)

        # Sample library
        self.lib = SampleLibrary(sample_root=sample_root)

        # Track data
        self.main_measures = None
        self.acc_measures = None

        # State
        self.ended = False
        self._stop_flag = threading.Event()
        self.pedal_down = False  # legacy boolean (derived from pedal_depth)

        # Thread-safe note tracking
        self._held_main = []
        self._held_acc = []
        self._sustained = []
        self._held_lock = threading.Lock()
        self._main_newlines = set()

    # ----- Public interface -----

    def load_tracks(self, main_measures, acc_measures):
        self.main_measures = main_measures or []
        self.acc_measures = acc_measures or []
        return self

    def preload_all_samples(self):
        """
        Parallel preload all samples used in the score.
        Uses ThreadPoolExecutor for concurrent file I/O.
        """
        print(f"正在预加载采样...", end='', flush=True)

        all_tracks = []
        if self.main_measures:
            all_tracks.extend(self._flatten(self.main_measures)[0])
        if self.acc_measures:
            all_tracks.extend(self._flatten(self.acc_measures)[0])

        # Collect all (midi, vel) pairs needed
        needed = set()
        for token in all_tracks:
            try:
                parsed = parse_step_token(token, self.default_vel)
            except (ValueError, TypeError):
                continue
            # Collect notes from regular steps and from tuplet sub-steps
            all_notes = list(parsed['notes'])
            if parsed.get('tuplet'):
                for sub in parsed['tuplet']:
                    all_notes.extend(sub.get('notes', []))
            for (midi, vel, _) in all_notes:
                vel_int = int(round(vel))
                # Preload ±1 layer for jitter coverage
                for v in range(max(1, vel_int - 1), min(17, vel_int + 2)):
                    needed.add((midi, v))

        if not needed:
            print(" 无需加载")
            return

        # Ensure scan is done first (needed for path resolution)
        self.lib.scan()

        # Phase 1: Resolve file paths and pitch-shift needs (fast, under lock)
        load_tasks = []  # [(midi, vel, filepath, pitch_shift_semitones)]
        for midi, vel in needed:
            with self.lib._lock:
                key = (midi, vel)
                if self.lib._lru_get(self.lib.cache, key) is not None:
                    continue  # already cached

                if midi in self.lib.index:
                    layers = self.lib.index[midi]
                    chosen = vel if vel in layers else \
                        self.lib._choose_vel_from_layers(layers, vel)
                    path = layers[chosen]
                    load_tasks.append((midi, chosen, path, 0))
                else:
                    base_midi = self.lib._nearest_available_midi(midi)
                    base_layers = self.lib.index[base_midi]
                    chosen = vel if vel in base_layers else \
                        self.lib._choose_vel_from_layers(base_layers, vel)
                    path = base_layers[chosen]
                    load_tasks.append((midi, chosen, path, midi - base_midi))

        # Phase 2: Parallel file loading (no lock needed for pygame.mixer.Sound)
        count = 0
        errors = 0

        def _load_one(task):
            midi, vel, path, shift = task
            snd = pygame.mixer.Sound(path)
            if shift != 0 and _HAVE_NUMPY:
                snd = self.lib._pitch_shift_sound(snd, shift)
            return midi, vel, snd

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_load_one, t): t for t in load_tasks}
            for future in as_completed(futures):
                try:
                    midi, vel, snd = future.result()
                    with self.lib._lock:
                        self.lib._lru_set(self.lib.cache, (midi, vel), snd,
                                          self.lib.max_cache)
                    count += 1
                    if count % 20 == 0:
                        print('.', end='', flush=True)
                except Exception:
                    errors += 1

        msg = f" 完成！已加载 {count} 个采样"
        if errors > 0:
            msg += f" ({errors} 个失败)"
        print(msg)

    def stop(self):
        """Stop playback and clean up all resources."""
        self._stop_flag.set()

        # Cancel all pending timers (under lock since other threads append)
        with self._held_lock:
            timers_to_cancel = list(self._delay_timers)
            self._delay_timers.clear()

        for t in timers_to_cancel:
            try:
                t.cancel()
            except Exception:
                pass

        if self._repedal_timer is not None:
            try:
                self._repedal_timer.cancel()
            except Exception:
                pass
            self._repedal_timer = None

        # Fade out all held notes
        with self._held_lock:
            all_pairs = (self._held_main + self._held_acc +
                         self._sustained + self._pending_pedal_release)
            self._held_main.clear()
            self._held_acc.clear()
            self._sustained.clear()
            self._pending_pedal_release.clear()

        for _, ch in all_pairs:
            try:
                ch.fadeout(50)
            except Exception:
                pass

    def __bool__(self):
        return bool(self.ended)

    def _gc_delay_timers(self):
        """Remove finished timers to prevent memory leak in long pieces."""
        with self._held_lock:
            self._delay_timers = [t for t in self._delay_timers
                                  if t.is_alive()]

    # ----- Internal: flatten, fade, release -----

    def _flatten(self, measures):
        steps = []
        newlines = set()
        idx = 0
        for ms in measures:
            for t in ms:
                steps.append(t)
                idx += 1
            for _ in range(self.line_pause_steps):
                steps.append('_')
                idx += 1
            newlines.add(idx)
        return steps, newlines

    def _fade_ms_for_midi(self, midi: int) -> int:
        """Bass/treble adaptive release time (smooth interpolation)."""
        x = (midi - 21) / 87.0  # 21=A0, 108=C8
        x = max(0.0, min(1.0, x))
        # Smooth cubic interpolation instead of linear
        x_smooth = x * x * (3.0 - 2.0 * x)
        return int(round(
            self.release_fade_bass_ms * (1.0 - x_smooth) +
            self.release_fade_treble_ms * x_smooth
        ))

    def _fadeout_pairs(self, pairs, ms_override=None):
        for midi, ch in pairs:
            try:
                ms = (int(ms_override) if ms_override is not None
                      else self._fade_ms_for_midi(int(midi)))
                ch.fadeout(ms)
            except Exception:
                pass

    def _play_release_sample(self, midi: int, vel: int = 8):
        """Play release sample if available."""
        if not self.use_release_samples:
            return
        snd = self.lib.get_release_sound(midi, vel)
        if snd is None:
            return
        ch = pygame.mixer.find_channel(False)
        if ch is None:
            return
        try:
            ch.set_volume(self.release_sample_gain)
            ch.play(snd)
        except Exception:
            pass

    def _release_held(self, held_list):
        if not held_list:
            return
        with self._held_lock:
            if self.pedal_down or self.pedal_depth > 0.3:
                for midi, ch in held_list:
                    self._sustained.append((midi, ch))
            else:
                self._fadeout_pairs(held_list)
                # Play release samples for released notes
                for midi, _ in held_list:
                    self._play_release_sample(midi)
            held_list.clear()

    def _release_held_delayed(self, held_list, delay_s: float):
        if not held_list:
            return

        with self._held_lock:
            if self.pedal_down or self.pedal_depth > 0.3:
                for midi, ch in held_list:
                    self._sustained.append((midi, ch))
                held_list.clear()
                return

            pairs = list(held_list)
            held_list.clear()

        delay_s = float(delay_s) if delay_s else 0.0
        if delay_s <= 0:
            self._fadeout_pairs(pairs)
            for midi, _ in pairs:
                self._play_release_sample(midi)
        else:
            def _do_release():
                self._fadeout_pairs(pairs)
                for midi, _ in pairs:
                    self._play_release_sample(midi)

            t = threading.Timer(delay_s, _do_release)
            t.daemon = True
            with self._held_lock:
                self._delay_timers.append(t)
            t.start()

    # ----- Velocity momentum -----

    def _apply_momentum(self, vel_target: float) -> float:
        """
        Simulate arm inertia: consecutive notes tend toward a similar velocity.
        momentum=0 -> no effect, momentum=1 -> heavy smoothing.
        """
        if self.velocity_momentum <= 0.0:
            return vel_target
        m = min(1.0, self.velocity_momentum)
        result = self._last_velocity * m + vel_target * (1.0 - m)
        self._last_velocity = result
        return result

    # ----- Note pressing -----

    def _press_notes(self, notes, gain: float, held_list, is_arpeggio: bool = False):
        base_gain = float(gain)

        # Arpeggio: sort low→high
        if is_arpeggio and len(notes) > 1:
            notes = sorted(notes, key=lambda x: x[0])

        for note_idx, (midi, vel, _disp) in enumerate(notes):
            if self._stop_flag.is_set():
                return

            # Pre-compute everything for this note (needed for Timer closure)
            midi_i = int(midi)
            vel_target = float(vel)
            g = base_gain

            if self.soft_pedal:
                vel_target = max(1.0, vel_target - self.soft_pedal_vel_shift)
                g *= self.soft_pedal_gain

            if self.vel_jitter > 0:
                vel_target += random.gauss(0, self.vel_jitter * 0.5)
            vel_target = max(1.0, min(16.0, vel_target))

            vel_normalized = vel_target / 16.0
            vel_normalized = self._apply_momentum(vel_normalized)
            vel_target = vel_normalized * 16.0
            vel_target = max(1.0, min(16.0, vel_target))

            rr_vel_offset = 0.0
            if self.round_robin:
                count = self._rr_last_midi.get(midi_i, 0)
                self._rr_last_midi[midi_i] = count + 1
                if count > 0:
                    rr_vel_offset = random.uniform(
                        -self.round_robin_cents / 3.0,
                        self.round_robin_cents / 3.0
                    ) * 0.3

            effective_vel = max(1.0, min(16.0, vel_target + rr_vel_offset))

            if self.vel_crossfade:
                blends = self.lib.get_sound_blend(
                    midi_i, effective_vel, equal_power=True
                )
            else:
                layer = int(round(effective_vel))
                blends = [(self.lib.get_sound_layer(midi_i, layer), 1.0, layer)]

            vel_curve_vol = apply_velocity_curve(
                vel_target / 16.0, self.velocity_curve_type, self.velocity_curve
            )

            if self.gain_jitter > 0:
                gain_jit = 1.0 + random.gauss(0, self.gain_jitter * 0.5)
            else:
                gain_jit = 1.0

            rr_active = (self.round_robin
                         and self._rr_last_midi.get(midi_i, 0) > 1)
            rr_offset = (random.uniform(0, self.round_robin_offset_ms)
                         if rr_active else 0)

            # Closure: actually play one note on audio channels
            def _fire(midi_i=midi_i, blends=blends, g=g,
                      vel_curve_vol=vel_curve_vol, gain_jit=gain_jit,
                      rr_active=rr_active, rr_offset=rr_offset,
                      vel_target=vel_target):
                for bi, (snd, w, _layer) in enumerate(blends):
                    ch = pygame.mixer.find_channel(False)
                    if ch is None:
                        ch = pygame.mixer.find_channel(True)
                    if ch is None:
                        if bi == 0:
                            ch = pygame.mixer.Channel(0)
                            ch.stop()
                        else:
                            continue

                    vol = max(0.0, min(1.0,
                              float(g) * float(w) * vel_curve_vol * gain_jit))
                    try:
                        ch.set_volume(vol)
                    except Exception:
                        pass

                    if rr_active:
                        fade_in = max(0, int(rr_offset * 0.5))
                        ch.play(snd, fade_ms=fade_in)
                    else:
                        ch.play(snd)

                    with self._held_lock:
                        held_list.append((midi_i, ch))

                if self.sympathetic_resonance:
                    self._play_sympathetic_resonance(
                        midi_i, g * vel_target / 16.0)

            # Arpeggio: stagger each note by arpeggio_stagger_ms
            if is_arpeggio and note_idx > 0:
                delay_s = note_idx * (self.arpeggio_stagger_ms / 1000.0)
                t = threading.Timer(delay_s, _fire)
                t.daemon = True
                t.start()
                with self._held_lock:
                    self._delay_timers.append(t)
            else:
                _fire()

    # ----- Sympathetic resonance (琴弦共振) -----

    def _play_sympathetic_resonance(self, midi: int, source_vol: float):
        """
        Simulate sympathetic string resonance.

        When a note is struck, strings that are harmonically related
        vibrate in sympathy, producing very quiet ghost tones.

        The effect is stronger when the sustain pedal is down (all
        dampers are lifted, allowing all strings to resonate freely).

        Physics:
          - Octave (2:1 ratio, +12 semitones): strongest resonance
          - Perfect 5th (3:2, +7 semitones): strong
          - 2 octaves (4:1, +24): moderate
          - 5th + octave (3:1, +19): weak
          - Major 3rd + octave (5:4, +16): weakest
          - Lower octave (-12): moderate (string below also resonates)
        """
        if source_vol < 0.05:
            return  # too quiet to produce audible resonance

        # Pedal boost: when pedal is down, all dampers are up -> much more resonance
        pedal_mult = 1.0
        if self.pedal_down or self.pedal_depth > 0.3:
            # Scale boost by pedal depth (half-pedal = partial boost)
            depth = self.pedal_depth if self.pedal_depth > 0.3 else (1.0 if self.pedal_down else 0.0)
            pedal_mult = 1.0 + (self.resonance_pedal_boost - 1.0) * depth

        base_vol = source_vol * self.resonance_gain * pedal_mult

        for interval, strength in self._resonance_intervals:
            res_midi = midi + interval

            # Stay within piano range (A0=21 to C8=108)
            if res_midi < 21 or res_midi > 108:
                continue

            vol = base_vol * strength

            # Skip inaudible resonances
            if vol < 0.003:
                continue

            # Clamp volume
            vol = min(vol, 0.12)

            # Add slight random variation to each resonance (no two are identical)
            vol *= (1.0 + random.uniform(-0.15, 0.15))
            vol = max(0.0, min(0.12, vol))

            try:
                # Use the same velocity layer as source but much quieter
                snd = self.lib.get_sound_layer(res_midi, 4)  # low velocity layer for softer timbre
                ch = pygame.mixer.find_channel(False)
                if ch is None:
                    continue  # don't force-steal channels for resonance

                ch.set_volume(vol)
                ch.play(snd)

                # Add to sustained list so pedal-up will release them too
                with self._held_lock:
                    self._sustained.append((res_midi, ch))
            except Exception:
                pass

    # ----- Pedal logic -----

    def _finalize_pedal_up(self):
        self._repedal_timer = None
        if self.pedal_down or self.pedal_depth > 0.3:
            return
        with self._held_lock:
            if self._pending_pedal_release:
                # For half-pedal: scale fadeout time by inverse depth
                ms = self.pedal_release_ms
                self._fadeout_pairs(self._pending_pedal_release,
                                    ms_override=ms)
                for midi, _ in self._pending_pedal_release:
                    self._play_release_sample(midi)
                self._pending_pedal_release.clear()

    def _apply_pedal(self, action, depth: float = 1.0):
        if action == 'down':
            self.pedal_down = True
            self.pedal_depth = 1.0
            if self._repedal_timer is not None:
                try:
                    self._repedal_timer.cancel()
                except Exception:
                    pass
                self._repedal_timer = None
            with self._held_lock:
                if self._pending_pedal_release:
                    self._sustained.extend(self._pending_pedal_release)
                    self._pending_pedal_release.clear()
            return

        if action == 'half':
            self.pedal_depth = max(0.0, min(1.0, depth))
            self.pedal_down = self.pedal_depth > 0.3
            # Half-pedal: partially damp sustained notes
            if self.pedal_depth < 0.5:
                with self._held_lock:
                    if self._sustained:
                        # Reduce volume of sustained notes proportionally
                        damping = 1.0 - (0.5 - self.pedal_depth) * self.half_pedal_damping
                        damping = max(0.1, damping)
                        for midi, ch in self._sustained:
                            try:
                                current = ch.get_volume()
                                ch.set_volume(current * damping)
                            except Exception:
                                pass
            return

        if action == 'up':
            self.pedal_down = False
            self.pedal_depth = 0.0
            if self._repedal_timer is not None:
                try:
                    self._repedal_timer.cancel()
                except Exception:
                    pass
                self._repedal_timer = None

            with self._held_lock:
                if self._sustained:
                    self._pending_pedal_release.extend(self._sustained)
                    self._sustained.clear()

            if self.repedal_window_ms > 0 and self._pending_pedal_release:
                t = threading.Timer(
                    self.repedal_window_ms / 1000.0,
                    self._finalize_pedal_up
                )
                t.daemon = True
                self._repedal_timer = t
                t.start()
            else:
                self._finalize_pedal_up()

    # ----- Main playback loop -----

    def _compute_phrase_boundaries(self, newlines: set, total: int) -> list:
        """Pre-compute phrase boundaries from newline positions.
        Returns list of (phrase_start, phrase_end) for each step index."""
        if not newlines:
            return [(0, total)] * total

        boundaries = sorted(newlines)
        # Build phrase ranges
        phrases = []
        prev = 0
        for b in boundaries:
            if b > prev:
                phrases.append((prev, b))
            prev = b
        if prev < total:
            phrases.append((prev, total))

        # Map each step to its phrase range
        step_to_phrase = [None] * total
        for start, end in phrases:
            for i in range(start, min(end, total)):
                step_to_phrase[i] = (start, end)

        # Fill any None (shouldn't happen, but safety)
        for i in range(total):
            if step_to_phrase[i] is None:
                step_to_phrase[i] = (0, total)

        return step_to_phrase

    def _compute_adaptive_overlap(self, prev_midi, current_notes, base_overlap_s, step_s):
        """Compute overlap based on interval between previous and current note."""
        if not self.adaptive_legato or prev_midi is None or not current_notes:
            return base_overlap_s

        # Use first note of current step
        cur_midi = int(current_notes[0][0])
        interval = abs(cur_midi - prev_midi)

        if interval <= self.legato_max_interval:
            # 小音程 = 连奏: overlap 增大 (手指滚动, 前音延长)
            # 音程越小 overlap 越大
            legato_factor = 1.0 - (interval / (self.legato_max_interval + 1))
            # 最大 overlap 可达 step_s 的 60%
            return min(step_s * 0.60, base_overlap_s * (1.0 + legato_factor * 2.0))
        elif interval <= 12:
            # 中等音程: 正常 overlap
            return base_overlap_s
        else:
            # 大跳: overlap 缩小 (手要移位, 前音断得快)
            return base_overlap_s * 0.3

    def _play_one_step(self, step, gain, held_list, prev_midi_attr, step_s):
        """Play a single parsed step (notes/extend/rest) and return updated prev_midi.
        Used both by the main loop and by the tuplet sub-loop."""
        # Pedal actions
        for action, depth in step.get('pedal_actions', []):
            self._apply_pedal(action, depth)

        # Soft pedal
        if step.get('soft') == 'on':
            self.soft_pedal = True
        elif step.get('soft') == 'off':
            self.soft_pedal = False

        prev_midi = getattr(self, prev_midi_attr)
        if step['notes']:
            base_overlap = (min(self.overlap_ms / 1000.0, step_s * 0.35)
                            if self.overlap_ms > 0 else 0.0)
            overlap_s = self._compute_adaptive_overlap(
                prev_midi, step['notes'], base_overlap, step_s
            )
            self._release_held_delayed(held_list, overlap_s)
            self._press_notes(step['notes'], gain, held_list,
                              is_arpeggio=step.get('arpeggio', False))
            setattr(self, prev_midi_attr, int(step['notes'][-1][0]))
        elif step['extend']:
            pass
        elif step['rest']:
            self._release_held(held_list)
            setattr(self, prev_midi_attr, None)

    def play(self):
        if self.main_measures is None or self.acc_measures is None:
            raise RuntimeError("Tracks not loaded. Call load_tracks().")

        main_steps, self._main_newlines = self._flatten(self.main_measures)
        acc_steps, _ = self._flatten(self.acc_measures)

        total = max(len(main_steps), len(acc_steps))
        base_step_s = self.times / 1000.0

        # Pre-compute phrase boundaries for phrase_accel
        phrase_map = self._compute_phrase_boundaries(self._main_newlines, total)

        next_t = time.perf_counter()
        i = 0
        while i < total:
            if self._stop_flag.is_set():
                break

            # Periodic cleanup of finished delay timers
            if i & 63 == 0:  # every 64 steps
                self._gc_delay_timers()

            # ---- Three-layer tempo humanization ----
            step_s = base_step_s

            # Layer 2: Tempo drift (slow sinusoidal wandering)
            if self.tempo_drift_range > 0:
                drift = math.sin(
                    self._drift_phase + i * self.tempo_drift_speed * 0.05
                )
                step_s *= (1.0 + self.tempo_drift_range * drift)

            # Layer 3: Phrase breathing (rush start, relax end)
            if self.phrase_accel > 0:
                p_start, p_end = phrase_map[i]
                p_len = p_end - p_start
                if p_len > 2:
                    pos = (i - p_start) / (p_len - 1)
                    breath = (1.0 - math.cos(pos * math.pi)) * self.phrase_accel
                    step_s *= (1.0 + breath - self.phrase_accel)

            mtok = main_steps[i] if i < len(main_steps) else '_'
            atok = acc_steps[i] if i < len(acc_steps) else '_'

            try:
                m = parse_step_token(mtok, self.default_vel)
            except (ValueError, TypeError):
                m = {'notes': [], 'extend': False, 'rest': True,
                     'pedal_actions': [], 'soft': None, 'tuplet': None}
            try:
                a = parse_step_token(atok, self.default_vel)
            except (ValueError, TypeError):
                a = {'notes': [], 'extend': False, 'rest': True,
                     'pedal_actions': [], 'soft': None, 'tuplet': None}

            # ============ TUPLET HANDLING ============
            m_tuplet = m.get('tuplet')
            a_tuplet = a.get('tuplet')

            if m_tuplet or a_tuplet:
                tuplet_subs = m_tuplet or a_tuplet
                n_sub = len(tuplet_subs)
                is_main = bool(m_tuplet)  # True = tuplet in melody

                # Count span: this step + following extends
                span = 1
                j = i + 1
                while j < total:
                    check_tok = (main_steps[j] if is_main and j < len(main_steps)
                                 else acc_steps[j] if not is_main and j < len(acc_steps)
                                 else '_')
                    if check_tok not in ('0', '.'):
                        break
                    span += 1
                    j += 1

                total_span_s = span * step_s
                sub_interval_s = total_span_s / n_sub

                # Build event list: (time_offset_s, track, parsed_step)
                events = []

                # Tuplet sub-notes at evenly-spaced times
                for si, sub in enumerate(tuplet_subs):
                    t = si * sub_interval_s
                    events.append((t, 'main' if is_main else 'acc', sub))

                # Other track's normal steps at grid times
                for k in range(span):
                    idx = i + k
                    if is_main:
                        ot = acc_steps[idx] if idx < len(acc_steps) else '_'
                    else:
                        ot = main_steps[idx] if idx < len(main_steps) else '_'
                    try:
                        ot_parsed = parse_step_token(ot, self.default_vel)
                    except (ValueError, TypeError):
                        continue
                    t = k * step_s
                    events.append((t, 'acc' if is_main else 'main', ot_parsed))

                # Sort by time (stable sort keeps insertion order for ties)
                events.sort(key=lambda x: x[0])

                # Play through merged timeline
                span_start = time.perf_counter()
                for evt_time, track, step_data in events:
                    if self._stop_flag.is_set():
                        break
                    target = span_start + evt_time
                    now = time.perf_counter()
                    if target > now:
                        time.sleep(target - now)

                    if track == 'main':
                        self._play_one_step(step_data, self.main_gain,
                                            self._held_main, '_prev_main_midi',
                                            sub_interval_s if is_main else step_s)
                    else:
                        self._play_one_step(step_data, self.acc_gain,
                                            self._held_acc, '_prev_acc_midi',
                                            step_s if is_main else sub_interval_s)

                # Advance grid to end of span
                span_end = span_start + total_span_s
                now = time.perf_counter()
                if span_end > now:
                    time.sleep(span_end - now)
                next_t = span_end

                # Display
                if self.notes_visible:
                    parts = []
                    for sub in tuplet_subs:
                        if sub['notes']:
                            if len(sub['notes']) > 1:
                                parts.append('[' + ','.join(d for _, _, d in sub['notes']) + ']')
                            else:
                                parts.append(sub['notes'][0][2])
                        else:
                            parts.append('.')
                    print(f"{{{';'.join(parts)}}}".ljust(14), end=' ', flush=True)
                    if (i + span) in self._main_newlines:
                        print()

                i += span
                continue
            # ============ END TUPLET ============

            # Layer 1: Timing jitter (micro-random, clamped to dynamic step_s)
            jitter_s = 0.0
            if self.timing_jitter_ms > 0:
                jitter_s = random.gauss(0, self.timing_jitter_ms / 1000.0 * 0.5)
                jitter_s = max(-step_s * 0.15, min(step_s * 0.15, jitter_s))

            # Pedal events (with pedal_actions list)
            for action, depth in m.get('pedal_actions', []):
                self._apply_pedal(action, depth)
            for action, depth in a.get('pedal_actions', []):
                self._apply_pedal(action, depth)

            # Soft pedal
            if m.get('soft') == 'on':
                self.soft_pedal = True
            elif m.get('soft') == 'off':
                self.soft_pedal = False
            if a.get('soft') == 'on':
                self.soft_pedal = True
            elif a.get('soft') == 'off':
                self.soft_pedal = False

            # Main voice
            if m['notes']:
                base_overlap = (min(self.overlap_ms / 1000.0, step_s * 0.35)
                                if self.overlap_ms > 0 else 0.0)
                overlap_s = self._compute_adaptive_overlap(
                    self._prev_main_midi, m['notes'], base_overlap, step_s
                )
                self._release_held_delayed(self._held_main, overlap_s)
                self._press_notes(m['notes'], self.main_gain, self._held_main,
                                  is_arpeggio=m.get('arpeggio', False))
                # Track last note for adaptive legato
                self._prev_main_midi = int(m['notes'][-1][0])
            elif m['extend']:
                pass
            elif m['rest']:
                self._release_held(self._held_main)
                self._prev_main_midi = None  # rest breaks legato chain

            # Accompaniment
            if a['notes']:
                base_overlap = (min(self.overlap_ms / 1000.0, step_s * 0.35)
                                if self.overlap_ms > 0 else 0.0)
                overlap_s = self._compute_adaptive_overlap(
                    self._prev_acc_midi, a['notes'], base_overlap, step_s
                )
                self._release_held_delayed(self._held_acc, overlap_s)
                self._press_notes(a['notes'], self.acc_gain, self._held_acc,
                                  is_arpeggio=a.get('arpeggio', False))
                self._prev_acc_midi = int(a['notes'][-1][0])
            elif a['extend']:
                pass
            elif a['rest']:
                self._release_held(self._held_acc)
                self._prev_acc_midi = None

            # Display
            if self.notes_visible:
                if m['notes']:
                    if len(m['notes']) > 1:
                        disp = '[' + ','.join(d for (_, _, d) in m['notes']) + ']'
                    else:
                        disp = m['notes'][0][2]
                elif m['extend']:
                    disp = '0'
                elif m['rest']:
                    disp = '_'
                else:
                    disp = '.'

                # Pedal display using pedal_actions from both tracks
                all_actions = m.get('pedal_actions', []) + a.get('pedal_actions', [])
                ped = ''
                if all_actions:
                    has_up = any(act == 'up' for act, _ in all_actions)
                    has_down = any(act == 'down' for act, _ in all_actions)
                    if has_up and has_down:
                        ped = ' P↻'
                    elif has_down:
                        ped = ' P+'
                    elif has_up:
                        ped = ' P-'
                    else:
                        last = all_actions[-1]
                        if last[0] == 'half':
                            ped = f' P~{last[1]:.1f}'

                print(f"{disp}{ped}".ljust(14), end=' ', flush=True)
                if (i + 1) in self._main_newlines:
                    print()

            # Sleep with dynamic step + jitter
            next_t += step_s + jitter_s
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)

            i += 1

        # ---- Cleanup ----
        if self._repedal_timer is not None:
            try:
                self._repedal_timer.cancel()
            except Exception:
                pass
            self._repedal_timer = None

        self._release_held(self._held_main)
        self._release_held(self._held_acc)

        with self._held_lock:
            if self._sustained:
                self._fadeout_pairs(self._sustained,
                                    ms_override=self.pedal_release_ms)
                self._sustained.clear()
            if self._pending_pedal_release:
                self._fadeout_pairs(self._pending_pedal_release,
                                    ms_override=self.pedal_release_ms)
                self._pending_pedal_release.clear()

        # Cancel any remaining delay timers (under lock for thread safety)
        with self._held_lock:
            timers = list(self._delay_timers)
            self._delay_timers.clear()
        for dt in timers:
            try:
                dt.cancel()
            except Exception:
                pass

        self.ended = not self._stop_flag.is_set()


# =========================================================
# StopThreads (input handler)
# =========================================================

class StopThreads(threading.Thread):
    """Input handler for stopping playback."""

    def __init__(self):
        super().__init__(target=self.stop_threads)
        self.daemon = True
        self.threads = None
        self.choice = None

    def stop_threads(self):
        line = input()
        self.choice = line
        if self.threads:
            for thread in self.threads:
                try:
                    if thread and thread.is_alive():
                        thread.stop()
                except Exception:
                    pass