import librosa
import pandas as pd
import numpy as np

def read_csv_file(csv_path, csv_column_list):
    # Open the CSV file for reading
    data = pd.read_csv(open(csv_path), sep="\t")#
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

def get_wav_duration(WaveDir_list):
    """
    Get wave's duration by librosa.get_duration(wave_path).
    * Input :
        wave_pathes : waves'path (list).
    * Output : 
        wave_times : waves'duration (list).
    """
    wave_times = []
    for wav in WaveDir_list:
        # wav_data, sample_rate = librosa.load(wav, sr=16000)
        wav_duration = librosa.get_duration(filename=wav) 
        wave_times.append(wav_duration)
    return wave_times




input_folder = './Data/Train_starGAN_v0/'

input_csv = input_folder + "combined.csv"
csv_column_list = ['wave_name','Speaker']
csv_data = read_csv_file(input_csv, csv_column_list)

speaker_set = set()
for spk in csv_data['Speaker']:
    if spk not in speaker_set:
        speaker_set.add(spk)
speakers = list(speaker_set)


speaker_name_list = []
wave_time_list = []
for speaker_name in speakers:
    WaveDir_list = []
    for wav_dir, spk_n in zip(csv_data['wave_name'], csv_data['Speaker']):
        if spk_n == speaker_name:
            WaveDir_list.append(wav_dir)
    wave_times = get_wav_duration(WaveDir_list)
    speaker_name_list.append(speaker_name)
    wave_time_list.append(np.sum(wave_times)/3600)
    print("Speaker: " + speaker_name + ", data time: " + str(np.sum(wave_times)/3600) + " hr.")


print("All wave's duration: " + str(np.sum(wave_time_list)) + " hr.")

save_path = input_folder + "wave_data_time.csv"
kwargs = {'Speaker': speaker_name_list, 'Wave_Times':wave_time_list}
save_csv_file(save_path, **kwargs)