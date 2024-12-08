import librosa.display
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 文件路径

    file1 = 'wavs/origin/ms_gt_5.wav'
    file2 = 'wavs/autovocoder/ms_gt_5_syn.wav'
    file3 = 'wavs/hifigan/ms_gt_5_generated.wav'

    # file1 = 'wavs/2.wav'
    # file2 = 'wavs/2_syn.wav'
    # file3 = 'wavs/2_generated-2.wav'

    # 读取音频文件
    y1, sr1 = librosa.load(file1)
    y2, sr2 = librosa.load(file2)
    y3, sr3 = librosa.load(file3)

    # 生成梅尔频谱图
    mel_spec1 = librosa.feature.melspectrogram(y=y1, sr=sr1)
    mel_spec2 = librosa.feature.melspectrogram(y=y2, sr=sr2)
    mel_spec3 = librosa.feature.melspectrogram(y=y3, sr=sr3)

    # 转换为对数频谱图
    log_mel_spec1 = librosa.power_to_db(mel_spec1, ref=np.max)
    log_mel_spec2 = librosa.power_to_db(mel_spec2, ref=np.max)
    log_mel_spec3 = librosa.power_to_db(mel_spec3, ref=np.max)

    # 绘制梅尔频谱图
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    librosa.display.specshow(log_mel_spec1, sr=sr1, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram of Origin')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(1, 3, 2)
    librosa.display.specshow(log_mel_spec2, sr=sr2, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram of AutoVocoder')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(1, 3, 3)
    librosa.display.specshow(log_mel_spec3, sr=sr3, x_axis='time', y_axis='mel')
    plt.title('Mel Spectrogram of Hifi-GAN')
    plt.colorbar(format='%+2.0f dB')

    plt.tight_layout()
    plt.savefig('ms_gt_5_mel_spectrogram.png')
    plt.show()


    def plot_mel_spectrograms(file_paths, fmin=85, fmax=255, n_mels=64):
        """
        生成多段音频中人类声音频率范围的梅尔频谱图，并横向排列

        :param file_paths: 音频文件路径列表
        :param sr: 采样率
        :param fmin: 最小频率 (人类声音起始频率)
        :param fmax: 最大频率 (人类声音终止频率)
        :param n_mels: 梅尔频率数量
        """
        y, sr = librosa.load(file_paths)

        # 计算梅尔频谱
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax
        )

        # 转换为对数梅尔频谱
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # 绘制梅尔频谱图
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', fmin=fmin, fmax=fmax, cmap='coolwarm'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (Human Voice Frequencies)')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()


    # plot_mel_spectrograms(file1)

    # 示例：绘制三段音频的梅尔频谱图
    # file_paths = [file1, file2, file3]
    # plot_mel_spectrograms(file_paths)
