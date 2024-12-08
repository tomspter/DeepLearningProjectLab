import os
import librosa
import numpy as np
from fastdtw import fastdtw
from pesq import pesq
from pesq import PesqError
from scipy.spatial.distance import euclidean


def resample_audio(audio_path, target_sr=16000):
    """
    Resample the audio to the target sampling rate using librosa.
    """
    y, sr = librosa.load(audio_path, sr=None)  # Load with original sample rate
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)  # Resample
    return y, target_sr


def calculate_pesq(reference_path, degraded_path, target_sr=16000):
    """
    Calculate PESQ score between the reference and degraded audio.
    """
    try:
        ref_audio, ref_sr = resample_audio(reference_path, target_sr)
        deg_audio, deg_sr = resample_audio(degraded_path, target_sr)

        # Ensure both audio files have the same length
        min_len = min(len(ref_audio), len(deg_audio))
        ref_audio = ref_audio[:min_len]
        deg_audio = deg_audio[:min_len]

        # Convert to 16-bit integer format
        ref_audio = (ref_audio * 32767).astype(np.int16)
        deg_audio = (deg_audio * 32767).astype(np.int16)

        # Calculate PESQ
        pesq_score = pesq(target_sr, ref_audio, deg_audio, 'wb')  # Wide-band PESQ
        return pesq_score
    except PesqError as e:
        print(f"PESQ Error: {e}")
        return None


def calculate_mcd(reference_path, degraded_path, n_mfcc=13, target_sr=16000):
    """
    Calculate Mel Cepstral Distortion (MCD) between reference and degraded audio.
    """
    ref_audio, _ = resample_audio(reference_path, target_sr)
    deg_audio, _ = resample_audio(degraded_path, target_sr)

    # Calculate MFCC features
    ref_mfcc = librosa.feature.mfcc(y=ref_audio, sr=target_sr, n_mfcc=n_mfcc).T
    deg_mfcc = librosa.feature.mfcc(y=deg_audio, sr=target_sr, n_mfcc=n_mfcc).T

    # 使用 DTW 对齐 MFCC 序列
    distance, path = fastdtw(ref_mfcc, deg_mfcc, dist=euclidean)

    # 计算 MCD
    mcd = (10 / np.log(10)) * np.sqrt((2 / n_mfcc) * distance)
    return mcd


def calculate_snr(reference_path, degraded_path, target_sr=16000):
    """
    Calculate Signal-to-Noise Ratio (SNR) between reference and degraded audio.
    """
    ref_audio, _ = resample_audio(reference_path, target_sr)
    deg_audio, _ = resample_audio(degraded_path, target_sr)

    # Ensure both audio files have the same length
    min_len = min(len(ref_audio), len(deg_audio))
    ref_audio = ref_audio[:min_len]
    deg_audio = deg_audio[:min_len]

    # Calculate signal power and noise power
    noise = ref_audio - deg_audio
    signal_power = np.mean(ref_audio ** 2)
    noise_power = np.mean(noise ** 2)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def main():
    folder_autovocoder = "wavs/autovocoder"
    folder_hifigan = "wavs/hifigan"
    folder_origin = "wavs/origin"
    target_sr = 16000  # Target sampling rate for PESQ, MCD, and SNR

    # Get file lists
    files_autovocoder = sorted(os.listdir(folder_autovocoder))
    files_hifigan = sorted(os.listdir(folder_hifigan))
    files_origin = sorted(os.listdir(folder_origin))

    # Initialize results and accumulators
    results_autovocoder = []
    results_hifigan = []
    total_pesq_autovocoder = 0
    total_mcd_autovocoder = 0
    total_snr_autovocoder = 0
    total_pesq_hifigan = 0
    total_mcd_hifigan = 0
    total_snr_hifigan = 0
    count_autovocoder = 0
    count_hifigan = 0

    # Compare autovocoder files with origin
    for file_autovocoder in files_autovocoder:
        prefix = '_'.join(file_autovocoder.split('_')[:3])  # Extract prefix
        matching_files = [f for f in files_origin if f.startswith(prefix)]

        if matching_files:
            ref_path = os.path.join(folder_origin, matching_files[0])
            deg_path = os.path.join(folder_autovocoder, file_autovocoder)

            pesq_score = calculate_pesq(ref_path, deg_path, target_sr)
            mcd_score = calculate_mcd(ref_path, deg_path, target_sr)
            snr_score = calculate_snr(ref_path, deg_path, target_sr)

            if pesq_score is not None and mcd_score is not None and snr_score is not None:
                results_autovocoder.append((file_autovocoder, matching_files[0], pesq_score, mcd_score, snr_score))
                total_pesq_autovocoder += pesq_score
                total_mcd_autovocoder += mcd_score
                total_snr_autovocoder += snr_score
                count_autovocoder += 1

    # Compare hifigan files with origin
    for file_hifigan in files_hifigan:
        prefix = '_'.join(file_hifigan.split('_')[:3])  # Extract prefix
        matching_files = [f for f in files_origin if f.startswith(prefix)]

        if matching_files:
            ref_path = os.path.join(folder_origin, matching_files[0])
            deg_path = os.path.join(folder_hifigan, file_hifigan)

            pesq_score = calculate_pesq(ref_path, deg_path, target_sr)
            mcd_score = calculate_mcd(ref_path, deg_path, target_sr)
            # snr_score = calculate_snr(ref_path, deg_path, target_sr)

            if pesq_score is not None and mcd_score is not None and snr_score is not None:
                results_hifigan.append((file_hifigan, matching_files[0], pesq_score, mcd_score, snr_score))
                total_pesq_hifigan += pesq_score
                total_mcd_hifigan += mcd_score
                total_snr_hifigan += snr_score
                count_hifigan += 1

    # Print results
    print("\n=== PESQ, MCD Results: Autovocoder ===")
    for ref_file, deg_file, pesq_score, mcd_score, snr_score in results_autovocoder:
        print(f"{ref_file} vs {deg_file}: PESQ = {pesq_score:.2f}, MCD = {mcd_score:.2f}")

    print("\n=== PESQ, MCD Results: Hifi-GAN ===")
    for ref_file, deg_file, pesq_score, mcd_score, snr_score in results_hifigan:
        print(f"{ref_file} vs {deg_file}: PESQ = {pesq_score:.2f}, MCD = {mcd_score:.2f}")

    # Calculate and print averages
    avg_pesq_autovocoder = total_pesq_autovocoder / count_autovocoder if count_autovocoder > 0 else 0
    avg_mcd_autovocoder = total_mcd_autovocoder / count_autovocoder if count_autovocoder > 0 else 0
    avg_snr_autovocoder = total_snr_autovocoder / count_autovocoder if count_autovocoder > 0 else 0
    avg_pesq_hifigan = total_pesq_hifigan / count_hifigan if count_hifigan > 0 else 0
    avg_mcd_hifigan = total_mcd_hifigan / count_hifigan if count_hifigan > 0 else 0
    avg_snr_hifigan = total_snr_hifigan / count_hifigan if count_hifigan > 0 else 0

    print(f"\nAverage PESQ score for Autovocoder: {avg_pesq_autovocoder:.2f}")
    print(f"Average MCD score for Autovocoder: {avg_mcd_autovocoder:.2f}")
    # print(f"Average SNR score for Autovocoder: {avg_snr_autovocoder:.2f}")
    print(f"Average PESQ score for Hifi-GAN: {avg_pesq_hifigan:.2f}")
    print(f"Average MCD score for Hifi-GAN: {avg_mcd_hifigan:.2f}")
    # print(f"Average SNR score for Hifi-GAN: {avg_snr_hifigan:.2f}")


if __name__ == "__main__":
    main()