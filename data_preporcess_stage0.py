import pandas as pd
import numpy as np
import os
from os import listdir
import glob

def read_csv_file(csv_path, csv_column_list):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path))#
    result_dict = dict()
    for column_name in csv_column_list:
        result_dict[column_name] = data[column_name]     
    return result_dict


def save_csv_file(save_path, **kwargs):
    column_names = list(kwargs.keys())
    data = list(kwargs.values())
    for idx in range(len(data)):
        if data[idx] == []:
            k_name = column_names[idx]
            print(f"{save_path=}")
            print(f"No data : {k_name}")
            kwargs[k_name] = [0]*len(data[0])

    csv_datalist= list(np.array(list(kwargs.values())).T)
    csv_data = pd.DataFrame(columns=column_names,data=csv_datalist)
    csv_data.to_csv(save_path, encoding='utf-8',index=False, sep='\t')        
    print(f"Already save csv file. {save_path}")
    return save_path    



# CSV --> Speaker

# input_csv = './Data/Raw_data/AI_Lab/AI_speech_data_use.csv'
# save_path = "Data/Train_starGAN_v0/csvs/AI_Lab.csv"

# csv_column_list = ['wave_name']
# csv_data = read_csv_file(input_csv, csv_column_list)

# speaker_list = []
# wave_paths = []
# for wav_path in csv_data['wave_name']:
#     speaker_name = wav_path.split('/')[0]
#     speaker_list.append(speaker_name)
#     wave_paths.append('Data/Raw_data/AI_Lab/' + wav_path)
# kwargs = {'wave_name':wave_paths, 'Speaker':speaker_list}
# save_csv_file(save_path, **kwargs)


# wave's folder --> CSV
input_folder = 'Data/Raw_data/AIshell_test/'
save_folder = "Data/AIshell_train_test/"
save_csv_name = 'test.csv'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    os.makedirs(save_folder + 'csvs/')
    os.makedirs(save_folder + 'waves/')
save_path = save_folder + 'csvs/' + save_csv_name 

spk_files = listdir(input_folder)

speaker_list = []
wav_path = []
for spk in spk_files:
    input_folder_dir = input_folder + spk + "/"
    wav_list = glob.glob(os.path.join(input_folder_dir, '*.wav'))
    spk_list = [spk]*len(wav_list)

    speaker_list += spk_list
    wav_path += wav_list

kwargs = {'wave_name':wav_path, 'Speaker':speaker_list}
save_csv_file(save_path, **kwargs)

