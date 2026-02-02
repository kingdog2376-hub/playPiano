# -*- coding: utf-8 -*-
"""
Enhanced playPiano.py - Production Grade
- 88-key piano (A0~C8) with international note names: C4, F#3, Bb2 ...
- Velocity layers with crossfade and humanization
- Sustain pedal, soft pedal (una corda), repedal window
- Preloading system to eliminate first-note latency
- Non-linear velocity curve for realistic dynamics
- Optimized for smooth playback and minimal latency
"""

import os
import re
import threading
import time
import random
import math
from collections import OrderedDict
from os import environ

environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

try:
    import numpy as np
    _HAVE_NUMPY = True
except Exception:
    np = None
    _HAVE_NUMPY = False


# -----------------------------
# Pygame audio init (safe/once)
# -----------------------------
_AUDIO_INIT_LOCK = threading.Lock()
_AUDIO_INITED = False

def init_audio(sample_rate: int = 44100, buffer: int = 512, channels: int = 2, num_mixer_channels: int = 256):
    """Initialize pygame + mixer once."""
    global _AUDIO_INITED
    if _AUDIO_INITED:
        return
    with _AUDIO_INIT_LOCK:
        if _AUDIO_INITED:
            return
        pygame.init()
        try:
            pygame.mixer.init(frequency=int(sample_rate), size=-16, channels=int(channels), buffer=int(buffer))
        except Exception:
            pygame.mixer.init()
        try:
            pygame.mixer.set_num_channels(int(num_mixer_channels))
        except Exception:
            pass
        _AUDIO_INITED = True


# -----------------------------
# Utilities: note name <-> MIDI
# -----------------------------

_NOTE_BASE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

def read_lines_auto_encoding(path: str):
    """Read text file lines with tolerant encoding detection."""
    data = open(path, "rb").read()
    for enc in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(enc).splitlines(True)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace").splitlines(True)

def note_name_to_midi(note_name: str) -> int:
    """Convert note name like C4, F#3, Bb2 to MIDI number."""
    m = re.fullmatch(r'([A-Ga-g])([#bs]?)([0-8])', note_name.strip())
    if not m:
        raise ValueError(f'Invalid note name: {note_name}')
    n, acc, octv = m.group(1).upper(), m.group(2), int(m.group(3))
    semi = _NOTE_BASE[n] + (1 if acc == '#' else -1 if acc == 'b' else 0)
    midi = (octv + 1) * 12 + semi
    return midi

def midi_to_note_name(midi: int) -> str:
    """Return a sharp-style name, e.g. 61 -> C#4."""
    if not (0 <= midi <= 127):
        raise ValueError('MIDI out of range')
    octv = midi // 12 - 1
    semi = midi % 12
    semi_to_name = {0:'C',1:'C#',2:'D',3:'D#',4:'E',5:'F',6:'F#',7:'G',8:'G#',9:'A',10:'A#',11:'B'}
    return f"{semi_to_name[semi]}{octv}"


# --------------------------------
# Sample library (Salamander-style)
# --------------------------------

class SampleLibrary:
    """
    Sample library with velocity layers and lazy/eager loading.
    Supports sparse sampling with pitch-shift fallback.
    """
    _SAMPLE_RE = re.compile(r'^(?P<n>[A-Ga-g])(?P<acc>[#bs]?)(?P<oct>[0-8])(?:[_-]?)v(?P<vel>\d{1,3})$')
    _A0_MIDI = 21
    _C8_MIDI = 108

    def __init__(self, sample_root: str = 'resources', allow_ext=('.wav', '.mp3', '.ogg')):
        self.sample_root = sample_root
        self.allow_ext = tuple(e.lower() for e in allow_ext)
        self.index: dict[int, dict[int, str]] = {}
        self.cache: 'OrderedDict[tuple[int,int], pygame.mixer.Sound]' = OrderedDict()
        self.max_cache = 4096

        self._available_midis: list[int] = []
        self._base_sound_cache: 'OrderedDict[tuple[int,int], pygame.mixer.Sound]' = OrderedDict()
        self._max_base_cache = 512

        self._scanned = False
        self._lock = threading.RLock()

    def scan(self):
        """Scan folder and build file index (doesn't load audio yet)."""
        if self._scanned:
            return
        for root, _, files in os.walk(self.sample_root):
            for fn in files:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in self.allow_ext:
                    continue
                stem = os.path.splitext(fn)[0]
                m = self._SAMPLE_RE.fullmatch(stem)
                if not m:
                    continue
                n = m.group('n').upper()
                acc = m.group('acc')
                if acc == 's':
                    acc = '#'
                octv = int(m.group('oct'))
                vel = int(m.group('vel'))
                try:
                    midi = note_name_to_midi(f"{n}{acc}{octv}")
                except ValueError:
                    continue
                if not (self._A0_MIDI <= midi <= self._C8_MIDI):
                    continue
                self.index.setdefault(midi, {})[vel] = os.path.join(root, fn)

        self._scanned = True
        self._available_midis = sorted(self.index.keys())

        if not self.index:
            raise FileNotFoundError(
                f"No piano samples found under '{self.sample_root}'. "
                f"Expected files like 'C4v8.wav' or 'Cs4v8.wav'."
            )

    def _cache_get(self, key):
        v = self.cache.get(key)
        if v is not None:
            try:
                self.cache.move_to_end(key)
            except Exception:
                pass
        return v

    def _cache_set(self, key, snd: pygame.mixer.Sound):
        self.cache[key] = snd
        try:
            self.cache.move_to_end(key)
        except Exception:
            pass
        while len(self.cache) > self.max_cache:
            try:
                self.cache.popitem(last=False)
            except Exception:
                break

    def _base_cache_get(self, key):
        v = self._base_sound_cache.get(key)
        if v is not None:
            try:
                self._base_sound_cache.move_to_end(key)
            except Exception:
                pass
        return v

    def _base_cache_set(self, key, snd: pygame.mixer.Sound):
        self._base_sound_cache[key] = snd
        try:
            self._base_sound_cache.move_to_end(key)
        except Exception:
            pass
        while len(self._base_sound_cache) > self._max_base_cache:
            try:
                self._base_sound_cache.popitem(last=False)
            except Exception:
                break

    def _choose_vel_from_layers(self, layers: dict[int, str], vel_req: int) -> int:
        vels = sorted(layers.keys())
        return min(vels, key=lambda v: abs(v - vel_req))

    def _nearest_available_midi(self, midi: int) -> int:
        if not self._available_midis:
            raise KeyError("Sample library is empty.")
        import bisect
        pos = bisect.bisect_left(self._available_midis, midi)
        if pos == 0:
            return self._available_midis[0]
        if pos >= len(self._available_midis):
            return self._available_midis[-1]
        left = self._available_midis[pos - 1]
        right = self._available_midis[pos]
        return left if abs(midi - left) <= abs(right - midi) else right

    def _pitch_shift_sound(self, base_snd: pygame.mixer.Sound, semitone_diff: int) -> pygame.mixer.Sound:
        if semitone_diff == 0:
            return base_snd
        if not _HAVE_NUMPY:
            raise RuntimeError(
                "Pitch-shift requires numpy. Install: pip install numpy"
            )
        arr = pygame.sndarray.array(base_snd)
        ratio = 2 ** (semitone_diff / 12.0)
        n = arr.shape[0]
        idx = np.arange(0, n - 1, ratio, dtype=np.float64)
        x = np.arange(n, dtype=np.float64)

        def interp_1d(y):
            out = np.interp(idx, x, y.astype(np.float64))
            out = np.clip(out, -32768, 32767)
            return out.astype(np.int16)

        if arr.ndim == 1:
            out = interp_1d(arr)
        else:
            chans = []
            for c in range(arr.shape[1]):
                chans.append(interp_1d(arr[:, c]))
            out = np.stack(chans, axis=1)
        return pygame.sndarray.make_sound(out)

    def _vel_pair(self, vels: list[int], vel_target: float) -> tuple[int, int, float]:
        if not vels:
            raise KeyError("No velocity layers available.")
        vels = sorted(vels)
        vt = float(vel_target)
        if vt <= vels[0]:
            return vels[0], vels[0], 0.0
        if vt >= vels[-1]:
            return vels[-1], vels[-1], 0.0
        import bisect
        pos = bisect.bisect_left(vels, vt)
        v2 = vels[pos]
        v1 = vels[pos - 1]
        if v2 == v1:
            return v1, v2, 0.0
        t = (vt - v1) / (v2 - v1)
        return v1, v2, max(0.0, min(1.0, t))

    def get_sound_layer(self, midi: int, vel_layer: int = 12) -> pygame.mixer.Sound:
        """Load and return a Sound for specific velocity layer."""
        self.scan()
        vel_layer = int(vel_layer)

        with self._lock:
            if midi in self.index:
                layers = self.index[midi]
                chosen = vel_layer if vel_layer in layers else self._choose_vel_from_layers(layers, vel_layer)
                key = (midi, chosen)
                snd0 = self._cache_get(key)
                if snd0 is not None:
                    return snd0
                snd = pygame.mixer.Sound(layers[chosen])
                self._cache_set(key, snd)
                return snd

            # Pitch-shift fallback
            base_midi = self._nearest_available_midi(midi)
            base_layers = self.index[base_midi]
            chosen = vel_layer if vel_layer in base_layers else self._choose_vel_from_layers(base_layers, vel_layer)

            key = (midi, chosen)
            snd0 = self._cache_get(key)
            if snd0 is not None:
                return snd0

            base_key = (base_midi, chosen)
            base_snd = self._base_cache_get(base_key)
            if base_snd is None:
                base_snd = pygame.mixer.Sound(base_layers[chosen])
                self._base_cache_set(base_key, base_snd)

            semis = midi - base_midi
            shifted = self._pitch_shift_sound(base_snd, semis)
            self._cache_set(key, shifted)
            return shifted

    def get_sound_blend(self, midi: int, vel_target: float = 12.0):
        """Return list of (Sound, weight, layer) for velocity crossfade."""
        self.scan()
        vt = float(vel_target)

        if midi in self.index:
            layers = self.index[midi]
        else:
            base_midi = self._nearest_available_midi(midi)
            layers = self.index[base_midi]

        vels = sorted(layers.keys())
        v1, v2, t = self._vel_pair(vels, vt)

        if v1 == v2 or t <= 0.0:
            return [(self.get_sound_layer(midi, v1), 1.0, v1)]
        if t >= 1.0:
            return [(self.get_sound_layer(midi, v2), 1.0, v2)]

        s1 = self.get_sound_layer(midi, v1)
        s2 = self.get_sound_layer(midi, v2)
        return [(s1, 1.0 - t, v1), (s2, t, v2)]

    def get_sound(self, midi: int, vel_req: int = 12) -> pygame.mixer.Sound:
        """Compatibility wrapper."""
        return self.get_sound_layer(midi, vel_req)


# ------------------------
# Score parsing/tokenizing
# ------------------------

def tokenize_score_line(line: str) -> list:
    """Split line into tokens, keeping brackets intact."""
    line = line.split('//', 1)[0].strip()
    if not line:
        return []
    tokens = []
    buf = []
    depth = 0
    for ch in line:
        if ch == '[':
            depth += 1
            buf.append(ch)
        elif ch == ']':
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


_DEGREE_TO_NOTE = {'1': 'C', '2': 'D', '3': 'E', '4': 'F', '5': 'G', '6': 'A', '7': 'B'}

def parse_note_atom(atom: str, default_vel: int = 12) -> tuple:
    """Parse note atom: C4, F#3, 1+, etc. Returns (midi, vel, display)."""
    atom = atom.strip()
    if not atom:
        raise ValueError("Empty note")
    
    # Legacy numeric degree
    m_deg = re.fullmatch(r'([1-7])([+-]{0,2})(?:[@vV]?(\d{1,3}))?', atom)
    if m_deg:
        deg = m_deg.group(1)
        shift = m_deg.group(2)
        vel_raw = m_deg.group(3)
        base_oct = 4
        oct_shift = shift.count('+') - shift.count('-')
        note_letter = _DEGREE_TO_NOTE[deg]
        note_name = f"{note_letter}{base_oct + oct_shift}"
        midi = note_name_to_midi(note_name)
        vel_req = default_vel
        if vel_raw:
            v = int(vel_raw)
            if 1 <= v <= 16:
                vel_req = v
            elif 1 <= v <= 127:
                vel_req = max(1, min(16, round(v / 127 * 16)))
        return midi, vel_req, note_name

    # Modern note name
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
    vel_req = default_vel
    if vel_raw:
        v = int(vel_raw)
        if 1 <= v <= 16:
            vel_req = v
        elif 1 <= v <= 127:
            vel_req = max(1, min(16, round(v / 127 * 16)))
        else:
            vel_req = default_vel
    display = f"{n}{acc}{octv}"
    return midi, vel_req, display


def parse_step_token(token: str, default_vel: int = 12) -> dict:
    """Parse time-step token into notes, pedal actions, etc."""
    token = token.strip()
    if not token:
        return {'notes': [], 'extend': False, 'rest': False, 'pedal': None, 'soft': None}

    parts = [p.strip() for p in token.split('|') if p.strip()]
    notes = []
    extend = False
    rest = False
    pedal = None
    soft = None

    for p in parts:
        p_upper = p.upper()
        if p in ('0',):
            extend = True
            continue
        if p in ('_', 'R', 'REST'):
            rest = True
            continue
        if p_upper in ('P+', 'PD', 'PEDALDOWN', 'PEDAL_DOWN', 'SUSTAINON', 'SUSTAIN_ON'):
            pedal = 'down'
            continue
        if p_upper in ('P-', 'PU', 'PEDALUP', 'PEDAL_UP', 'SUSTAINOFF', 'SUSTAIN_OFF'):
            pedal = 'up'
            continue
        if p_upper in ('U+', 'UC+', 'UNACORDAON', 'UNACORDA_ON', 'SOFTPEDALON', 'SOFTPEDAL_ON', 'SOFTON', 'SOFT_ON'):
            soft = 'on'
            continue
        if p_upper in ('U-', 'UC-', 'UNACORDAOFF', 'UNACORDA_OFF', 'SOFTPEDALOFF', 'SOFTPEDAL_OFF', 'SOFTOFF', 'SOFT_OFF'):
            soft = 'off'
            continue

        if p.startswith('[') and p.endswith(']'):
            inside = p[1:-1].strip()
            if inside:
                atoms = [a for a in re.split(r'[,\s]+', inside) if a]
                for a in atoms:
                    notes.append(parse_note_atom(a, default_vel))
            continue

        notes.append(parse_note_atom(p, default_vel))

    if notes:
        extend = False
        rest = False
    return {'notes': notes, 'extend': extend, 'rest': rest, 'pedal': pedal, 'soft': soft}


# ----------------
# Notes file loader
# ----------------

class Notes:
    """Load score files from directory."""
    def __init__(self):
        self.notes = OrderedDict()

    def load_note(self, file_path: str):
        file = os.path.basename(file_path)
        suffix = file.split('.')[-1]
        title = file.split('_')[0]
        times = int(file[file.find('_') + 1:file.rfind('.')])
        lines = read_lines_auto_encoding(file_path)
        measures = [tokenize_score_line(line) for line in lines]
        if self.notes.get(title) is None:
            self.notes[title] = {}
        self.notes[title][suffix] = measures
        self.notes[title]['times'] = times

    def load_notes(self, path: str):
        for root, _, files in os.walk(path):
            for file in files:
                self.load_note(os.path.join(root, file))
        return self.notes


# -------------------------
# Piano Sequencer
# -------------------------

class PianoSequencer(threading.Thread):
    """Production-grade piano sequencer with preloading and realistic dynamics."""

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
        # audio
        sample_rate: int = 44100,
        audio_buffer: int = 512,
        num_mixer_channels: int = 256,
        # timing
        line_pause_steps: int = 0,
        # damping
        release_fade_bass_ms: int | None = None,
        release_fade_treble_ms: int | None = None,
        pedal_release_ms: int | None = None,
        # pedals
        soft_pedal_vel_shift: int = 2,
        soft_pedal_gain: float = 0.88,
        # realism
        vel_crossfade: bool = True,
        vel_jitter: float = 0.35,
        gain_jitter: float = 0.02,
        repedal_window_ms: int = 85,
        # NEW: velocity curve (Gemini's suggestion)
        velocity_curve: float = 1.8
    ):
        super().__init__(target=self.play)
        self.times = int(times)
        self.notes_visible = notes_visible
        self.default_vel = default_vel
        self.main_gain = float(main_gain)
        self.acc_gain = float(acc_gain)
        self.overlap_ms = int(overlap_ms)

        init_audio(sample_rate=sample_rate, buffer=audio_buffer, channels=2, num_mixer_channels=num_mixer_channels)

        self.line_pause_steps = max(0, int(line_pause_steps))

        # Damping
        self.release_fade_ms = int(release_fade_ms)
        self.release_fade_bass_ms = int(release_fade_bass_ms) if release_fade_bass_ms is not None else max(60, int(self.release_fade_ms * 1.6))
        self.release_fade_treble_ms = int(release_fade_treble_ms) if release_fade_treble_ms is not None else max(40, int(self.release_fade_ms * 0.9))
        self.pedal_release_ms = int(pedal_release_ms) if pedal_release_ms is not None else max(60, int(min(self.release_fade_ms, 120)))

        # Pedals
        self.soft_pedal = False
        self.soft_pedal_vel_shift = int(soft_pedal_vel_shift)
        self.soft_pedal_gain = float(soft_pedal_gain)

        # Realism
        self.vel_crossfade = bool(vel_crossfade)
        self.vel_jitter = float(vel_jitter)
        self.gain_jitter = float(gain_jitter)
        self.velocity_curve = float(velocity_curve)  # NEW: non-linear volume

        # Repedal
        self.repedal_window_ms = int(repedal_window_ms)
        self._pending_pedal_release = []
        self._repedal_timer = None
        self._delay_timers = []

        self.lib = SampleLibrary(sample_root=sample_root)
        self.main_measures = None
        self.acc_measures = None

        self.ended = 0
        self._stop_flag = threading.Event()

        self.pedal_down = False
        self._held_main = []
        self._held_acc = []
        self._sustained = []
        self._main_newlines = set()

    def load_tracks(self, main_measures, acc_measures):
        self.main_measures = main_measures or []
        self.acc_measures = acc_measures or []
        return self

    def preload_all_samples(self):
        """
        CRITICAL: Preload all samples used in the score to eliminate first-note latency.
        This is Gemini's #1 fix and is absolutely necessary.
        """
        print(f"正在预加载采样 (步长 {self.times}ms)...", end='', flush=True)
        
        all_tracks = []
        if self.main_measures:
            all_tracks.extend(self._flatten(self.main_measures)[0])
        if self.acc_measures:
            all_tracks.extend(self._flatten(self.acc_measures)[0])

        count = 0
        loaded_keys = set()

        for token in all_tracks:
            parsed = parse_step_token(token, self.default_vel)
            for (midi, vel, _) in parsed['notes']:
                # Handle velocity jitter range: preload ±1 layer
                for v in range(max(1, vel - 1), min(17, vel + 2)):
                    key = (midi, v)
                    if key not in loaded_keys:
                        try:
                            self.lib.get_sound_layer(midi, v)
                            loaded_keys.add(key)
                            count += 1
                            if count % 15 == 0:
                                print('.', end='', flush=True)
                        except Exception:
                            pass
        
        print(f" 完成！已加载 {count} 个采样")

    def stop(self):
        self._stop_flag.set()
        for t in self._delay_timers:
            try:
                t.cancel()
            except:
                pass
        self._delay_timers.clear()
        if self._repedal_timer is not None:
            try:
                self._repedal_timer.cancel()
            except:
                pass
            self._repedal_timer = None

    def __bool__(self):
        return self.ended == 1

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
        lo, hi = 21, 108
        x = (midi - lo) / (hi - lo)
        x = max(0.0, min(1.0, x))
        return int(round(self.release_fade_bass_ms * (1.0 - x) + self.release_fade_treble_ms * x))

    def _fadeout_pairs(self, pairs, ms_override: int | None = None):
        for midi, ch in pairs:
            try:
                ms = int(ms_override) if ms_override is not None else self._fade_ms_for_midi(int(midi))
                ch.fadeout(ms)
            except Exception:
                pass

    def _release_held(self, held_list):
        if not held_list:
            return
        if self.pedal_down:
            for midi, ch in held_list:
                self._sustained.append((midi, ch))
        else:
            self._fadeout_pairs(held_list)
        held_list.clear()

    def _release_held_delayed(self, held_list, delay_s: float):
        if not held_list:
            return
        if self.pedal_down:
            for midi, ch in held_list:
                self._sustained.append((midi, ch))
            held_list.clear()
            return

        pairs = list(held_list)
        held_list.clear()

        delay_s = float(delay_s) if delay_s else 0.0
        if delay_s <= 0:
            self._fadeout_pairs(pairs)
        else:
            t = threading.Timer(delay_s, self._fadeout_pairs, args=(pairs,))
            t.daemon = True
            self._delay_timers.append(t)
            t.start()

    def _press_notes(self, notes, gain: float, held_list):
        base_gain = float(gain)
        for midi, vel, _disp in notes:
            if self._stop_flag.is_set():
                return

            midi_i = int(midi)
            vel_target = float(vel)
            g = base_gain

            if self.soft_pedal:
                vel_target = max(1.0, vel_target - float(self.soft_pedal_vel_shift))
                g = g * self.soft_pedal_gain

            if self.vel_jitter > 0:
                vel_target += random.uniform(-self.vel_jitter, self.vel_jitter)
            vel_target = max(1.0, min(16.0, vel_target))

            if self.vel_crossfade:
                blends = self.lib.get_sound_blend(midi_i, vel_target)
            else:
                layer = int(round(vel_target))
                blends = [(self.lib.get_sound_layer(midi_i, layer), 1.0, layer)]

            for i, (snd, w, _layer) in enumerate(blends):
                ch = pygame.mixer.find_channel(False)
                if ch is None:
                    ch = pygame.mixer.find_channel(True)
                if ch is None:
                    if i == 0:
                        ch = pygame.mixer.Channel(0)
                        ch.stop()
                    else:
                        continue

                vol = float(g) * float(w)

                if self.gain_jitter > 0:
                    vol *= (1.0 + random.uniform(-self.gain_jitter, self.gain_jitter))
                vol = max(0.0, min(1.0, vol))

                # NEW: Apply non-linear velocity curve (Gemini's suggestion)
                # This makes dynamics more realistic (exponential, not linear)
                if self.velocity_curve != 1.0:
                    vol = math.pow(vol, self.velocity_curve)

                try:
                    ch.set_volume(vol)
                except Exception:
                    pass
                ch.play(snd)
                held_list.append((midi_i, ch))

    def _finalize_pedal_up(self):
        self._repedal_timer = None
        if self.pedal_down:
            return
        if self._pending_pedal_release:
            self._fadeout_pairs(self._pending_pedal_release, ms_override=self.pedal_release_ms)
            self._pending_pedal_release.clear()

    def _apply_pedal(self, action):
        if action == 'down':
            self.pedal_down = True
            if self._repedal_timer is not None:
                try:
                    self._repedal_timer.cancel()
                except Exception:
                    pass
                self._repedal_timer = None
            if self._pending_pedal_release:
                self._sustained.extend(self._pending_pedal_release)
                self._pending_pedal_release.clear()
            return

        if action == 'up':
            self.pedal_down = False
            if self._repedal_timer is not None:
                try:
                    self._repedal_timer.cancel()
                except Exception:
                    pass
                self._repedal_timer = None

            if self._sustained:
                self._pending_pedal_release.extend(self._sustained)
                self._sustained.clear()

            if self.repedal_window_ms > 0 and self._pending_pedal_release:
                t = threading.Timer(self.repedal_window_ms / 1000.0, self._finalize_pedal_up)
                t.daemon = True
                self._repedal_timer = t
                t.start()
            else:
                self._finalize_pedal_up()

    def play(self):
        if self.main_measures is None or self.acc_measures is None:
            raise RuntimeError("Tracks not loaded. Call load_tracks().")

        main_steps, self._main_newlines = self._flatten(self.main_measures)
        acc_steps, _ = self._flatten(self.acc_measures)

        total = max(len(main_steps), len(acc_steps))
        step_s = self.times / 1000.0

        next_t = time.perf_counter()
        for i in range(total):
            if self._stop_flag.is_set():
                break

            mtok = main_steps[i] if i < len(main_steps) else '_'
            atok = acc_steps[i] if i < len(acc_steps) else '_'

            m = parse_step_token(mtok, self.default_vel)
            a = parse_step_token(atok, self.default_vel)

            if m['pedal']:
                self._apply_pedal(m['pedal'])
            if a['pedal']:
                self._apply_pedal(a['pedal'])

            if m.get('soft') == 'on':
                self.soft_pedal = True
            elif m.get('soft') == 'off':
                self.soft_pedal = False
            if a.get('soft') == 'on':
                self.soft_pedal = True
            elif a.get('soft') == 'off':
                self.soft_pedal = False

            if m['notes']:
                overlap_s = min(self.overlap_ms / 1000.0, step_s * 0.35) if self.overlap_ms > 0 else 0.0
                self._release_held_delayed(self._held_main, overlap_s)
                self._press_notes(m['notes'], self.main_gain, self._held_main)
            elif m['extend']:
                pass
            elif m['rest']:
                self._release_held(self._held_main)

            if a['notes']:
                overlap_s = min(self.overlap_ms / 1000.0, step_s * 0.35) if self.overlap_ms > 0 else 0.0
                self._release_held_delayed(self._held_acc, overlap_s)
                self._press_notes(a['notes'], self.acc_gain, self._held_acc)
            elif a['extend']:
                pass
            elif a['rest']:
                self._release_held(self._held_acc)

            if self.notes_visible:
                if m['notes']:
                    disp = '[' + ','.join(d for (_,_,d) in m['notes']) + ']' if len(m['notes']) > 1 else m['notes'][0][2]
                elif m['extend']:
                    disp = '0'
                elif m['rest']:
                    disp = '_'
                else:
                    disp = '.'
                ped = ' P+' if (m['pedal']=='down' or a['pedal']=='down') else (' P-' if (m['pedal']=='up' or a['pedal']=='up') else '')
                # Gemini's optimization: add flush=True to prevent buffering
                print(f"{disp}{ped}".ljust(12), end=' ', flush=True)
                if (i + 1) in self._main_newlines:
                    print()

            next_t += step_s
            dt = next_t - time.perf_counter()
            if dt > 0:
                time.sleep(dt)

        # Cleanup
        if self._repedal_timer is not None:
            try:
                self._repedal_timer.cancel()
            except Exception:
                pass
            self._repedal_timer = None

        self._release_held(self._held_main)
        self._release_held(self._held_acc)

        if self._sustained:
            self._fadeout_pairs(self._sustained, ms_override=self.pedal_release_ms)
            self._sustained.clear()

        if self._pending_pedal_release:
            self._fadeout_pairs(self._pending_pedal_release, ms_override=self.pedal_release_ms)
            self._pending_pedal_release.clear()

        self.ended = 1


class StopThreads(threading.Thread):
    """Input handler for stopping playback."""
    def __init__(self):
        super().__init__(target=self.stop_threads)
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
