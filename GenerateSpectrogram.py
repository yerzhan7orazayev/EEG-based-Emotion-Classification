import os

import numpy as np
from matplotlib.pyplot import specgram

path = '/home/yerzhan/Desktop/Fall 2018/Project/'
os.chdir(path)

loaded = np.load('data_python/s01.npz')
data = loaded['obj1'][:, :, 384:8064]
labels = loaded['obj2']
print(data.shape)

valence_labels = labels[:,0].reshape((40, 1))
y_valence = valence_labels.copy()
y_valence[valence_labels > 5] = 1
y_valence[valence_labels <= 5] = 0

segment_length = 512
segment_amount = int(7680/segment_length)
spect_data = np.zeros((40*32*segment_amount, 17, 10))
aug_label = np.zeros((40*32*segment_amount, 1))
print(spect_data.shape)

count = 0
for trail in range(40):
    for channel in range(32):
        for segment in range(segment_amount): 
            segment_ind = range(segment*segment_length, (segment+1)*segment_length)
            #print(segment_ind)
            x = data[trail, channel, segment_ind]
            spectogram = specgram(x, NFFT = 32, Fs = 128, noverlap = 8)[0]
            #print(spectogram.shape)
            spect_data[count, :, :] = spectogram
            aug_label[count,:] = y_valence[trail]
            count = count + 1
            del x
            del spectogram
            if count % 500 == 0:
                print("Count is " + str(count))

path = '/home/yerzhan/Desktop/Fall 2018/Project/data_python/spect_data_512.npz'
np.savez(path, data = spect_data, label = aug_label)
