import librosa
import numpy as np
import os
import random

#track, sr = librosa.load('...mp3')
#mel = librosa.feature.melspectrogram(track, sr=sr, n_mels=128)

# todo: associate

path = './fma_small'

files = os.walk('./fma_small')

min_samples = 10000000

i = 0

complete = ['033', '028', '001', '094', '048', '135', '141', '043', '127', '126', '036', '078', '027', '130', '088', '096', '025', '113', '108', '140', '134', '052', '008', '047', '024', '049', '075', '152', '074', '079', '046', '116', '022', '057', '003', '030', '076', '112', '026', '090', '107', '129', '005', '021', '100', '110', '012', '016', '083', '145', '019', '085', '053', '035', '056', '121', '138', '014', '050', '068', '153', '118', '122', '132', '017', '058', '150', '105', '149', '142', '139', '077', '007', '061', '066', '034', '086', '032', '020', '037', '055', '114', '011', '106', '119', '128', '071', '151', '062' ]

with open('./fma_metadata/melspectrogram.csv', 'a') as f:
    for folder in files:
        if folder[0] != path and folder[0].split('/')[2] not in complete:
            print(folder[0].split('/')[2])
            for track in folder[2]:
                print('{}/{}'.format(folder[0], track))
                clip, sr = librosa.load('{}/{}'.format(folder[0], track))
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
