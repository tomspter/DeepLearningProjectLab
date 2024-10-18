import os

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # dir_path = '../hifi-gan/LJSpeech-1.1/wavs'
    dir_path = 'wavs'
    audio_files = [f for f in os.listdir(dir_path) if f.endswith('.wav')]

    # 检查音频文件数量
    print(f"Number of audio files: {len(audio_files)}")

    audio_durations = []
    for file in audio_files:
        y, sr = librosa.load(os.path.join(dir_path, file), sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        audio_durations.append(duration)

    print(f"Average audio duration: {np.mean(audio_durations)} seconds")
    print(f"Max audio duration: {np.max(audio_durations)} seconds")
    print(f"Min audio duration: {np.min(audio_durations)} seconds")

    # Number of audio files: 13100
    # Average audio duration: 6.573822616883904 seconds
    # Max audio duration: 10.096190476190475 seconds
    # Min audio duration: 1.1100680272108843 seconds


    y, sr = librosa.load(os.path.join(dir_path, audio_files[0]), sr=None)

    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform of Audio File')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig('waveform.png')
    plt.close()

    # STFT
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Audio File')
    plt.savefig('spectrogram.png')
    plt.close()

    # 2. 计算梅尔频谱图
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    # 3. 转换为分贝单位
    S_dB = librosa.power_to_db(S, ref=np.max)

    # 4. 绘制梅尔频谱图
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()


