
import os
from scipy.io import wavfile
from pydub import AudioSegment

from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import pandas as pd
import random





def split(sound):
    dBFS = sound.dBFS
    chunks = split_on_silence(sound,
        min_silence_len = 100,
        silence_thresh = dBFS-16,
        keep_silence = 100
    )
    return chunks

def combine(_src):
    audio = AudioSegment.empty()
    for i,filename in enumerate(os.listdir(_src)):
        if filename.endswith('.wav'):
            filename = os.path.join(_src, filename)
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


# VCTK Corpus Path
__CORPUSPATH__ = "./Raw_data/ZH-TW"

# output path
__OUTPATH__ = "./Data"

speakers = [200,201,202,203]



# downsample to 24 kHz
for p in speakers:
    directory = __OUTPATH__ + '/' + str(p)
    if not os.path.exists(directory):
        audio = combine(__CORPUSPATH__ + '/' + str(p))
        chunks = split(audio)
        save_chunks(chunks, directory)


# get all speakers
data_list = []
for path, subdirs, files in os.walk(__OUTPATH__):
    for name in files:
        if name.endswith(".wav"):
            speaker = int(path.split('/')[-1].replace('p', ''))
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
text_file = open(__OUTPATH__ + "/train_list.txt", "w")
text_file.write(file_str)
text_file.close()

file_str = ""
for index, k in test_data.iterrows():
    file_str += k['Path'] + "|" + str(k['Speaker'] - 1) + '\n'
text_file = open(__OUTPATH__ + "/val_list.txt", "w")
text_file.write(file_str)
text_file.close()