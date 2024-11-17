# written by: Mohammed Salah Al-Radhi <malradhi@tmit.bme.hu>
# at BME-TMIT, Budapest, 28-30 March 2023
# https://github.com/hcy71o/AutoVocoder



from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from models import Generator, Encoder
from torch.utils.data import DistributedSampler, DataLoader
from complexdataset import ComplexDataset, mel_spectrogram, get_dataset_filelist, MAX_WAV_VALUE, load_wav
from stft import TorchSTFT
import matplotlib.pyplot as plt
import pickle

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict



def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):

    encoder = Encoder(h).to(device)
    state_dict_e = load_checkpoint(a.checkpoint_encoder_file, device)
    encoder.load_state_dict(state_dict_e['encoder'])
    encoder.eval()
    print("Complete loading encoder")
    

    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(a.checkpoint_generator_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    print("Complete loading generator")   


    filelist = os.listdir(a.input_wavs_dir)

    test_filelist = []
    for file in os.listdir(a.input_wavs_dir):
        f = os.path.join(a.input_wavs_dir, file)
        test_filelist += [f]
        

    validset = ComplexDataset(test_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=False,
                              data_path=a.input_wavs_dir)
    validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)



    with torch.no_grad():
            for j, batch in enumerate(validation_loader):
                print(batch)
                
                x, y, filename, y_mel = batch

                print("Starting with = ", filename)

                print('x = ', x.shape)
                l = encoder(x.to(device))
                print('l = ', l.shape)


                mel = l.cpu().numpy().squeeze()

#                 # save mel files
#                 with open(filename[0][:-4] + '.mel', 'wb') as handle:
#                     pickle.dump(mel, handle)

#                 print(filename[0] + '.mel')


#                 # load mel files
#                 with open(filename[0][:-4] + '.mel', 'rb') as handle:
#                     mel_data = pickle.load(handle)


                x = torch.from_numpy(np.expand_dims(mel, axis=0)).to(device).float()
                y_g_hat = generator(x)

                audio = y_g_hat.squeeze()
                audio = audio * MAX_WAV_VALUE
                audio = audio.cpu().numpy().astype('int16')
                
#                audio2= np.mean(audio)
#                audio = audio/audio2
                
                 
                output_path = os.path.join(filename[0][:-4] + "_syn.wav")
                write(output_path, h.sampling_rate, audio)
                print("All samples are saved here: ", output_path)
                   
                print("Done ...!") 


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_generator_file', default='/root/autodl-tmp/DP_Group/Autovocoder/professor/g_00210000')
    parser.add_argument('--checkpoint_encoder_file',default='/root/autodl-tmp/DP_Group/Autovocoder/professor/e_00210000')
    parser.add_argument('--input_training_file',default='/home/malradhi/AutoVocoder/LJSpeech-1.1/training.txt')
    parser.add_argument('--data_path', default='test_files')
    parser.add_argument('--input_wavs_dir',default='test_files')
    a = parser.parse_args()

    config_file ='/root/autodl-tmp/DP_Group/Autovocoder/professor/config.json'
    print(config_file)
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

