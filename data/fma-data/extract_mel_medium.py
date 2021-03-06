import librosa
import os
import zipfile as zf
import random
import numpy as np
# faster, but segfaults:
# import torchaudio

csv_str = ''

count = 0

with zf.ZipFile('./fma_medium.zip') as fma_zip:
    contents = fma_zip.namelist()
    for track in contents:
        if '.mp3' in track:
            track_path = fma_zip.extract(track)
            
            count += 1

            print("Extracting track #{}, {}".format(count, track_path.split('/')[-1]), sep=' ', end='\r', flush=True)
            
            try:
                # faster, but segfaults
                """
                #clip = torchaudio.load(track_path)[0].numpy()
                #clip = librosa.to_mono(np.transpose(clip))
                """

                clip, _ = librosa.load(track_path)
                
                os.remove(track_path)
                
                mel = librosa.feature.melspectrogram(clip, n_mels=128)
                
                if mel.shape[1] > 129:
                    # Write header - track id
                    csv_str += '{}'.format(int(track_path.split('.')[0].split('/')[-1]))
                    for x in range(127):
                        csv_str += ','
                    csv_str += '\n'
                    # Write data for that track; 128 lines of 1024 elements
                    for freq in range(len(mel)):
                        r = random.randint(0, mel.shape[1]-128)
                        csv_str += '{}'.format(mel[freq][r])
                        for sample in mel[freq][r+1:r+128]:
                            csv_str += ',{}'.format(sample)
                        csv_str += '\n'
            except:
                print('Bad Track! {}'.format(track_path))

os.remove('./fma_medium.zip')

with zf.ZipFile('./fma_metadata/melspectrogram_medium.zip', 'w') as mel_zip:
    mel_zip.writestr('melspectrogram.csv', csv_str)
