
import os
import glob
from scipy.io import wavfile
from pydub import AudioSegment
import numpy as np
from pydub import AudioSegment
from pydub.silence import split_on_silence
import pandas as pd
import random
import csv

# speakers' list to dic 
def list2dic(keys):
    values = np.arange(keys.__len__())
    dic = dict(zip(keys, values))
    return dic


def split(sound):
    dBFS = sound.dBFS
    chunks = split_on_silence(sound,
        min_silence_len = 100,
        silence_thresh = dBFS-16,
        keep_silence = 100
    )
    return chunks

def combine_audio(wav_path_list):
    audio = AudioSegment.empty()
    for i,filename in enumerate(wav_path_list):
        audio += AudioSegment.from_wav(filename)
    return audio

def save_chunks(chunks, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    counter = 0

    target_length = 5 * 1000
    output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) < target_length:
            output_chunks[-1] += chunk
        else:
            # if the last output chunk is longer than the target length,
            # we can start a new one
            output_chunks.append(chunk)

    for chunk in output_chunks:
        chunk = chunk.set_frame_rate(24000)
        chunk = chunk.set_channels(1)
        counter = counter + 1
        chunk.export(os.path.join(directory, str(counter) + '.wav'), format="wav")

def read_csv_file(csv_path, csv_column_list):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path), sep="\t")#
    result_dict = dict()
    for column_name in csv_column_list:
        result_dict[column_name] = data[column_name]     
    return result_dict

def combine_csv_files(input_csv_folder, output_folder):
    filenames = sorted(glob.glob(input_csv_folder + "/" + "*.csv"))
    print(filenames)
    if len(filenames) > 1 :
        combined_csv = pd.concat( [ pd.read_csv(f) for f in filenames ] )   
    elif len(filenames) == 1:
        combined_csv = pd.read_csv(filenames[0])
        print("Only one csv to combine !!")
    else:
        print("No csv to combine !!")
        combined_csv = pd.DataFrame(0, index=np.arange(10), columns=['wave_name','labels','wave_times'])
    combined_csv_name = output_folder + "combined" + ".csv"
    combined_csv.to_csv(combined_csv_name, index=False )
    print(f"{combined_csv_name} = 合併資料夾內的csv檔案")


input_folder = './Data/AIshell_train_test/'

# Combine csvs
combine_csv_files(input_folder + "csvs/", input_folder)


input_csv = input_folder + "combined.csv"
# input_csv = input_folder + "Xi_5sec.csv"
csv_column_list = ['wave_name','Speaker']
csv_data = read_csv_file(input_csv, csv_column_list)

speaker_set = set()
for spk in csv_data['Speaker']:
    if spk not in speaker_set:
        speaker_set.add(spk)
speakers = list(speaker_set)


print(speakers)
speakers_dic = list2dic(speakers)
print(speakers_dic)
with open(input_folder + 'speakers_dct.csv', 'w') as f:  
    writer = csv.writer(f)
    for k, v in speakers_dic.items():
       writer.writerow([k, v])


# downsample to 24 kHz
for speaker_name in speakers:
    save_directory = input_folder + "waves/" + speaker_name
    wave_dirs = []
    for wav_dir, spk_n in zip(csv_data['wave_name'], csv_data['Speaker']):
        if spk_n == speaker_name:
            wave_dirs.append(wav_dir)
    audio = combine_audio(wave_dirs)
    chunks = split(audio)
    save_chunks(chunks, save_directory)


__OUTPATH__ = input_folder + 'waves/'
# get all speakers
data_list = []
for path, subdirs, files in os.walk(__OUTPATH__):
    for name in files:
        if name.endswith(".wav"):
            speaker = path.split('/')[-1]
            if speaker in speakers:
                data_list.append({"Path": os.path.join(path, name), "Speaker": int(speakers.index(speaker)) + 1})


data_list = pd.DataFrame(data_list)
data_list = data_list.sample(frac=1)


split_idx = round(len(data_list) * 0.1)

test_data = data_list[:split_idx]
train_data = data_list[split_idx:]


# write to file 
file_str = ""
for index, k in train_data.iterrows():
    file_str += k['Path'] + "|" +str(k['Speaker'] - 1)+ '\n'
    # print(str(k['Speaker'] - 1))
text_file = open(input_folder + "train_list.txt", "w")
text_file.write(file_str)
text_file.close()

file_str = ""
for index, k in test_data.iterrows():
    file_str += k['Path'] + "|" + str(k['Speaker'] - 1) + '\n'
text_file = open(input_folder + "val_list.txt", "w")
text_file.write(file_str)
text_file.close()

