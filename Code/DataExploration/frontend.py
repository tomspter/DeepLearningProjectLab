import asyncio
import glob
import os
import shutil
import subprocess
import time

import gradio as gr
import librosa
import librosa.display
import numpy as np
import soundfile as sf
from PIL import Image
from matplotlib import pyplot as plt


def delete_file(file_path):
    files = glob.glob(f"{file_path}/*")
    for file in files:
        try:
            os.remove(file)
        except Exception as e:
            return f"Error deleting files: {e}"


def run_py_file(file_path):
    try:
        result = subprocess.run(
            ["python", file_path],
            capture_output=True,
            text=True,
            check=True
        )
        return f"Inference Output:\n{result.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Error running inference script:\n{e.stderr}"

async def run_py_file_async(file_path):
    try:
        # 异步启动子进程
        process = await asyncio.create_subprocess_exec(
            "python",
            file_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # 异步等待脚本执行完成
        stdout, stderr = await process.communicate()
        if process.returncode == 0:
            return f"Inference Output:\n{stdout.decode()}"
        else:
            return f"Error running inference script:\n{stderr.decode()}"
    except Exception as e:
        return f"Unexpected error occurred: {str(e)}"


def generate_mel_spectrogram(wav_path, mel_output_path):
    """Generate a Mel spectrogram plot for the given .wav file."""
    audio, sr = librosa.load(wav_path, sr=22050)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # plt.figure(figsize=(10, 4))
    # librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis="time", y_axis="mel")
    # plt.colorbar(format="%+2.0f dB")
    # plt.title("Mel Spectrogram")
    # plt.tight_layout()

    plt.figure(figsize=(10, 4))
    # plt.imshow(mel_db, aspect='auto', origin='lower', cmap='viridis')
    plt.imshow(mel_db, aspect='auto', origin='lower', cmap='magma',
               extent=[0, mel_db.shape[1] * (len(audio) / sr) / mel_db.shape[1], 0, sr / 2])
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()

    plt.savefig(mel_output_path)
    plt.close()

def delete_directory(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)  # Recursively delete the directory
        print(f"Deleted directory: {dir_path}")
    else:
        print(f"Directory {dir_path} does not exist.")



async def save_audio_with_librosa(audio):
    if audio is None:
        return ["No audio input provided.", None, None, None, None, None, None, None, None]
    print(audio)
    delete_file("/root/autodl-tmp/DP_Group/hifi-gan/test_files")
    delete_file("/root/autodl-tmp/DP_Group/Autovocoder/test_files")

    y, sr = librosa.load(audio, sr=None)
    y = librosa.resample(y, orig_sr=sr, target_sr=22050)

    timestamp = time.strftime("%Y%m%d%H%M%S")
    input_save_path = f"/root/autodl-tmp/DP_Group/origin_wavs/{timestamp}.wav"

    sf.write(input_save_path, y, 22050,format='WAV',subtype='PCM_16')
    sf.write(f"/root/autodl-tmp/DP_Group/Autovocoder/test_files/{timestamp}.wav", y, 22050,format='WAV',subtype='PCM_16')
    sf.write(f"/root/autodl-tmp/DP_Group/hifi-gan/test_files/{timestamp}.wav", y, 22050,format='WAV',subtype='PCM_16')

    origin_mel_path = f"/root/autodl-tmp/DP_Group/origin_wavs/{timestamp}_mel.png"
    generate_mel_spectrogram(input_save_path, origin_mel_path)

    path_to_delete = '/root/autodl-tmp/DP_Group/Autovocoder/test_files/.ipynb_checkpoints'
    delete_directory(path_to_delete)
    path_to_delete2 = '/root/autodl-tmp/DP_Group/hifi-gan/test_files/.ipynb_checkpoints'
    delete_directory(path_to_delete2)

    result1, result2 = await asyncio.gather(
        run_py_file_async("Autovocoder/Inference_malradhi.py"),
        run_py_file_async("hifi-gan/inference.py")
    )
    # result1 = run_py_file("Autovocoder/Inference_malradhi.py")
    print(result1)
    # result2 = run_py_file("hifi-gan/inference.py")
    print(result2)

    autovocoder_wav_path = f"/root/autodl-tmp/DP_Group/Autovocoder/generated_files/{timestamp}_syn.wav"
    hifigan_wav_path = f"/root/autodl-tmp/DP_Group/hifi-gan/generated_files/{timestamp}_generated.wav"
    if not os.path.exists(autovocoder_wav_path) or not os.path.exists(hifigan_wav_path):
        return f"Generated files not found for timestamp: {timestamp}."

    autovocoder_mel_path = f"/root/autodl-tmp/DP_Group/Autovocoder/generated_files/{timestamp}_autovocoder_mel.png"
    hifigan_mel_path = f"/root/autodl-tmp/DP_Group/hifi-gan/generated_files/{timestamp}_hifigan_mel.png"
    generate_mel_spectrogram(autovocoder_wav_path, autovocoder_mel_path)
    generate_mel_spectrogram(hifigan_wav_path, hifigan_mel_path)

    return [
        input_save_path,
        origin_mel_path,
        autovocoder_wav_path,
        autovocoder_mel_path,
        hifigan_wav_path,
        hifigan_mel_path,
    ]


def compare_images(image1, image2, image3):
    # 如果任意图片为空，返回默认占位图
    default_image = Image.new("RGB", (200, 200), color="gray")
    images = [
        Image.open(image1) if image1 else default_image,
        Image.open(image2) if image2 else default_image,
        Image.open(image3) if image3 else default_image
    ]

    # 统一调整尺寸，拼接图片
    # resized_images = [img.resize((300, 300)) for img in images]
    combined_image = np.hstack([np.array(img) for img in images])
    return Image.fromarray(combined_image)

with gr.Blocks() as iface:
    with gr.Row():
        audio_input = gr.Microphone(label="Record your voice", type="filepath")
    submit_button = gr.Button("Submit")
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Original Audio**")
            audio_origin = gr.Audio(label="Original Audio")
            image_origin = gr.Image(label="Original Mel Spectrogram",interactive=True,type="filepath")
        with gr.Column():
            gr.Markdown("**Autovocoder**")
            audio_auto = gr.Audio(label="Autovocoder Audio")
            image_auto = gr.Image(label="Autovocoder Mel Spectrogram",interactive=True,type="filepath")
        with gr.Column():
            gr.Markdown("**HiFi-GAN**")
            audio_hifi = gr.Audio(label="HiFi-GAN Audio")
            image_hifi = gr.Image(label="HiFi-GAN Mel Spectrogram",interactive=True,type="filepath")
    compare_button = gr.Button("compare")
    with gr.Row():
        comparison_image = gr.Image(label="compare", interactive=False)
    submit_button.click(
        fn=save_audio_with_librosa,
        inputs= audio_input,
        outputs= [audio_origin,image_origin,audio_auto,image_auto,audio_hifi,image_hifi]
    )

    compare_button.click(
        fn=compare_images,
        inputs=[image_origin, image_auto, image_hifi],
        outputs=comparison_image
    )

# 启动界面
iface.launch(server_name="0.0.0.0")
