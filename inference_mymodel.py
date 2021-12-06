import random
import yaml
from munch import Munch
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import librosa

from Utils.ASR.models import ASRCNN
from Utils.JDC.model import JDCNet
from models import Generator, MappingNetwork, StyleEncoder

import time
import soundfile as sf
import os
import pandas as pd

to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300)
mean, std = -4, 4

def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor

def build_model(model_params={}):
    args = Munch(model_params)
    generator = Generator(args.dim_in, args.style_dim, args.max_conv_dim, w_hpf=args.w_hpf, F0_channel=args.F0_channel)
    mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
    style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
    
    nets_ema = Munch(generator=generator,
                     mapping_network=mapping_network,
                     style_encoder=style_encoder)

    return nets_ema

def compute_style(speaker_dicts):
    reference_embeddings = {}
    for key, (path, speaker) in speaker_dicts.items():
        if path == "":
            label = torch.LongTensor([speaker]).to('cuda')
            latent_dim = starganv2.mapping_network.shared[0].in_features
            ref = starganv2.mapping_network(torch.randn(1, latent_dim).to('cuda'), label)
        else:
            wave, sr = librosa.load(path, sr=24000)
            audio, index = librosa.effects.trim(wave, top_db=30)
            if sr != 24000:
                wave = librosa.resample(wave, sr, 24000)
            mel_tensor = preprocess(wave).to('cuda')

            with torch.no_grad():
                label = torch.LongTensor([speaker])
                ref = starganv2.style_encoder(mel_tensor.unsqueeze(1), label)
        reference_embeddings[key] = (ref, label)
    
    return reference_embeddings

def read_csv_file(csv_path, csv_column_list):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path), sep="\t")#
    result_dict = dict()
    for column_name in csv_column_list:
        result_dict[column_name] = data[column_name]     
    return result_dict

# starGAN_model_path = 'Models/VCTK20/epoch_00150.pth'
# starGAN_model_config = 'Models/VCTK20/config.yml'

# speakers = [225,228,229,230,231,233,236,239,240,244,226,227,232,243,254,256,258,259,270,273]
# speaker_list = [273, 259, 258, 243, 254, 244, 236, 233, 230, 228]
# speaker_wav_folder = 'Demo/VCTK-corpus/p'
# speaker_wav_name = '_023.wav'

# starGAN_model_path = 'Models/ZH4_test/epoch_00120.pth'
# starGAN_model_config = 'Models/ZH4_test/config.yml'
# speakers = [200,201,202,203]
# speaker_list = [200,201,202,203]
# speaker_wav_folder = 'Demo/Chinese_corpus/p'
# speaker_wav_name = '_1.wav'

#input_folder = './Data/Train_starGAN_test/'
#input_csv = input_folder + "combined.csv"
#csv_column_list = ['wave_name','Speaker']
#csv_data = read_csv_file(input_csv, csv_column_list)

#speaker_set = set()
#for spk in csv_data['Speaker']:
#    if spk not in speaker_set:
#        speaker_set.add(spk)
#speakers = list(speaker_set)
#print("Speakers: " + ",".join(speakers))

starGAN_model_path = 'Models/Train_starGAN_test/epoch_00150.pth'
starGAN_model_config = 'Models/Train_starGAN_test/config.yml'
# speaker_list = ['Xi_v1','Xi_v2','伊薇','劉于碩','易辰','臧思齊','謝佳']
speaker_list = ['p225','p226','p227','p228','p229','p230']
speaker_wav_folder = 'Demo/test/'
speaker_wav_name = '11.wav'

source_wave = 'Data/p256/1.wav'
# source_wave =  'Demo/VCTK-corpus/p233/p233_023.wav'
# source_wave = '/home/c95hcw/ASR_Data/Dataset/raw_data/waves/speaker_data/易辰/record_10.wav'
output_folder = 'Result/Train_starGAN_test/' 
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
output_name = 'Epo150_demo1_'





# load F0 model
F0_model = JDCNet(num_class=1, seq_len=192)
params = torch.load("Utils/JDC/bst.t7")['net']
F0_model.load_state_dict(params)
_ = F0_model.eval()
F0_model = F0_model.to('cuda')

# load vocoder
from parallel_wavegan.utils import load_model
vocoder = load_model("Models/Vocoder/checkpoint-400000steps.pkl").to('cuda').eval()
vocoder.remove_weight_norm()
_ = vocoder.eval()

# load starganv2
model_path = starGAN_model_path

with open(starGAN_model_config) as f:
    starganv2_config = yaml.safe_load(f)
starganv2 = build_model(model_params=starganv2_config["model_params"])
params = torch.load(model_path, map_location='cpu')
params = params['model_ema']
_ = [starganv2[key].load_state_dict(params[key]) for key in starganv2]
_ = [starganv2[key].eval() for key in starganv2]
starganv2.style_encoder = starganv2.style_encoder.to('cuda')
starganv2.mapping_network = starganv2.mapping_network.to('cuda')
starganv2.generator = starganv2.generator.to('cuda')


# Target Speaker (with reference, using style encoder)
selected_speakers = speaker_list
speaker_dicts = {}
for s in selected_speakers:
    k = s
    # speaker_dicts['p' + str(s)] = (speaker_wav_folder + str(k) + '/p' + str(k) + speaker_wav_name, speakers.index(s))
    speaker_dicts[str(s)] = (speaker_wav_folder + str(k) + '/' + speaker_wav_name, speakers.index(s))
reference_embeddings = compute_style(speaker_dicts)


# load input wave
wav_path = source_wave
audio, source_sr = librosa.load(wav_path, sr=24000)
audio = audio / np.max(np.abs(audio))
audio.dtype = np.float32


# Conversion
# Convert by style encoder
start = time.time()
    
source = preprocess(audio).to('cuda:3')
keys = []
converted_samples = {}
reconstructed_samples = {}
converted_mels = {}

for key, (ref, _) in reference_embeddings.items():
    with torch.no_grad():
        f0_feat = F0_model.get_feature_GAN(source.unsqueeze(1))
        out = starganv2.generator(source.unsqueeze(1), ref, F0=f0_feat)

        c = out.transpose(-1, -2).squeeze().to('cuda')
        y_out = vocoder.inference(c)
        y_out = y_out.view(-1).cpu()
        sf.write(output_folder + output_name + str(key) + '.wav', y_out, 24000)
end = time.time()
print('total processing time: %.3f sec' % (end - start) )
