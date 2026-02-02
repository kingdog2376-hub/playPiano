# -*- coding: utf-8 -*-
"""
Enhanced main.py - Production Grade
- Fixed interaction deadlock (cross-platform solution)
- Added preloading system to eliminate first-note latency
- Clearer user prompts
"""

import os
import time
from playPiano import PianoSequencer, Notes, StopThreads


if __name__ == '__main__':
    # Cross-platform path
    playlist = Notes().load_notes(os.path.join('resources', 'notes'))

    stop_threads = StopThreads()
    
    while True:
        print('\n' + '='*60)
        print('请选择您想弹奏的琴谱：')
        for idx, (title, song) in enumerate(playlist.items()):
            print('{:3}. {}'.format(idx + 1, title.ljust(28, '-')), end='' if (idx + 1) % 3 else '\n')

        print('\n' + '='*60)
        print('请输入要弹奏的歌曲编号 (播放时按回车停止)：', end='')
        
        # Fixed interaction logic (Gemini's issue #2)
        # Check if previous song finished vs user stopped it
        if stop_threads.is_alive():
            stop_threads.join()
            choice = stop_threads.choice
        else:
            choice = input()

        # Empty input means continue (show menu again)
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
        notes = song.get('notes', [])
        accompaniments = song.get('accompaniments', [])
        times = song.get('times', 180)

        print(f'\n🎹 正在准备演奏: {title} (步长 {times}ms)')
        
        # Create player with optimized settings
        player = PianoSequencer(
            times=times,
            sample_root='resources',
            notes_visible=True,
            main_gain=1.0,
            acc_gain=0.75,
            release_fade_ms=140,
            overlap_ms=60,
            # Realism features
            vel_crossfade=True,
            vel_jitter=0.35,
            gain_jitter=0.02,
            repedal_window_ms=85,
            velocity_curve=1.8,  # NEW: Non-linear dynamics (Gemini's suggestion)
            # Audio
            num_mixer_channels=384,
            audio_buffer=512
        ).load_tracks(notes, accompaniments)

        # CRITICAL: Preload samples (Gemini's fix #1)
        # This eliminates first-note latency and "pops"
        try:
            player.preload_all_samples()
        except Exception as e:
            print(f'⚠️  预加载出错: {e}')
            print('将尝试继续播放（可能有延迟）...')

        print('▶️  开始播放... (按回车键停止)\n')
        player.start()

        # Start input listener
        if not stop_threads.is_alive():
            stop_threads = StopThreads()
        stop_threads.threads = [player]
        stop_threads.start()

        # Wait for playback to finish
        player.join()
        
        # Give user feedback
        if player.ended:
            print('\n✅ 播放完毕！')
        else:
            print('\n⏹️  已停止播放')
        
        # Small pause before showing menu again
        time.sleep(0.5)
