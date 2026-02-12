# -*- coding: utf-8 -*-
"""
Enhanced main.py - Production Grade
- All new features enabled: continuous velocity, timing jitter,
  velocity momentum, half-pedal, release samples, S-curve dynamics
"""

import os
import time
from playPiano import PianoSequencer, Notes, StopThreads, CURVE_EXPONENTIAL


if __name__ == '__main__':
    playlist = Notes().load_notes(os.path.join('resources', 'notes'))

    if not playlist:
        print('❌ 未找到任何乐谱文件，请检查 resources/notes 目录')
        exit(1)

    stop_threads = StopThreads()

    while True:
        print('\n' + '=' * 60)
        print('请选择您想弹奏的琴谱：')
        for idx, (title, song) in enumerate(playlist.items()):
            print('{:3}. {}'.format(idx + 1, title.ljust(28, '-')),
                  end='' if (idx + 1) % 3 else '\n')

        print('\n' + '=' * 60)
        print('请输入要弹奏的歌曲编号 (播放时按回车停止)：', end='')

        if stop_threads.is_alive():
            stop_threads.join()
            choice = stop_threads.choice
        else:
            choice = input()

        if choice is None or str(choice).strip() == '':
            continue

        try:
            choice = int(choice)
            if not (1 <= choice <= len(playlist)):
                raise ValueError
        except (ValueError, TypeError):
            print('❌ 输入无效，请重新输入！')
            continue

        title, song = list(playlist.items())[choice - 1]
        # Support various file extensions: .notes/.accompaniments or .txt etc.
        notes = song.get('notes', [])
        accompaniments = song.get('accompaniments', song.get('acc', []))
        times = song.get('times', 180)

        if not notes and not accompaniments:
            # Fallback: use first non-meta key as notes track
            for k, v in song.items():
                if k != 'times' and isinstance(v, list) and v:
                    if not notes:
                        notes = v
                    elif not accompaniments:
                        accompaniments = v

        if not notes and not accompaniments:
            print('❌ 该乐谱没有有效的音轨数据')
            continue

        print(f'\n🎹 正在准备演奏: {title} (步长 {times}ms)')

        player = PianoSequencer(
            times=times,
            sample_root='resources',
            notes_visible=True,
            main_gain=1.0,
            acc_gain=0.75,
            release_fade_ms=140,
            overlap_ms=60,
            # ===== Velocity & dynamics =====
            vel_crossfade=True,          # 等功率交叉淡入 (equal-power)
            vel_jitter=0.35,             # 力度随机 (gaussian)
            gain_jitter=0.02,            # 音量微扰
            velocity_curve=1.8,          # 曲线强度
            velocity_curve_type=CURVE_EXPONENTIAL,  # 指数曲线
            # ===== Humanization =====
            timing_jitter_ms=1.0,        # 微时序抖动 ±3ms
            velocity_momentum=0.07,      # 手臂惯性 (0=无, 1=重)
            # ===== Pedal =====
            repedal_window_ms=85,        # 换踏板窗口
            half_pedal_damping=0.5,      # 半踩踏板阻尼系数
            # ===== Sympathetic resonance (琴弦共振) =====
            sympathetic_resonance=False,   # 开启琴弦共振
            resonance_gain=0.028,         # 共振音量 (2.8% of source)
            resonance_pedal_boost=2.5,    # 踏板踩下时共振增强 2.5 倍
            # ===== Tempo humanization (三层 Rubato) =====
            tempo_drift_range=0,       # 整体速度漂移 ±4%
            tempo_drift_speed=0,        # 漂移速率 (慢呼吸)
            phrase_accel=0.,            # 乐句呼吸 6% (句首微赶,句尾微拖)
            # ===== Round Robin (同音重复变化) =====
            round_robin=False,             # 同音重复时微变音色
            round_robin_cents=1.0,        # 音高偏移 ±3 cents
            round_robin_offset_ms=4.0,    # 起始点偏移 ±8ms
            # ===== Arpeggio (琶音) =====
            arpeggio_stagger_ms=35.0,     # 琶音每音间隔 35ms
            # ===== Adaptive legato (自适应连奏) =====
            adaptive_legato=False,         # 根据音程动态调 overlap
            legato_max_interval=4,        # ≤4半音视为连奏
            # ===== Release =====
            use_release_samples=True,    # 释音采样 (如果有的话)
            release_sample_gain=0.3,     # 释音音量
            # ===== Audio engine =====
            num_mixer_channels=384,
            audio_buffer=512
        ).load_tracks(notes, accompaniments)

        # Parallel preload
        try:
            player.preload_all_samples()
        except Exception as e:
            print(f'⚠️  预加载出错: {e}')
            print('将尝试继续播放（可能有延迟）...')

        print('▶️  开始播放... (按回车键停止)\n')
        player.start()

        if not stop_threads.is_alive():
            stop_threads = StopThreads()
        stop_threads.threads = [player]
        stop_threads.start()

        player.join()

        if player.ended:
            print('\n✅ 播放完毕！')
        else:
            print('\n⏹️  已停止播放')

        time.sleep(0.5)