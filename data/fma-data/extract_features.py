import librosa
import numpy as np
import os
import random
import shutil

#track, sr = librosa.load('...mp3')
#mel = librosa.feature.melspectrogram(track, sr=sr, n_mels=128)

# todo: associate

path = './fma_small'

files = os.walk('./fma_small')

min_samples = 10000000

i = 0

with open('./fma_metadata/melspectrogram.csv', 'a') as f:
    for folder in files:
        if folder[0] != path
            print(folder[0].split('/')[2])
            for track in folder[2]:
                print('{}/{}'.format(folder[0], track))
                clip, sr = librosa.load('{}/{}'.format(folder[0], track))
                clip = librosa.to_mono(clip)
                try:
                    mel = librosa.feature.melspectrogram(clip, sr=sr, n_mels=128)
                    if min_samples > mel.shape[1]:
                        min_samples = mel.shape[1]
                    
                    if mel.shape[1] > 129:
                        # Write header - track id
                        f.write('{}'.format(int(track.split('.')[0])))
                        for x in range(127):
                            f.write(',')
                        f.write('\n')
                    # Write data for that track; 128 lines of 1024 elements
                        for freq in range(len(mel)):
                            r = random.randint(0, mel.shape[1]-128)
                            f.write('{}'.format(mel[freq][r]))
                            for sample in mel[freq][r+1:r+128]:
                                f.write(',{}'.format(sample))
                            f.write('\n')
            
                except:
                    print('Bad Track! {}'.format(track))
                    print(min_samples)

print(min_samples)
