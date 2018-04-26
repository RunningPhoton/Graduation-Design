import matplotlib.pyplot as plt
import numpy as np


lstm_sizes = [128, 192, 256, 384, 512, 600, 768]
dir = '01/'
start = 4
one_epoch = 80
mean_size = 5
offset = 5

def get_X_Y(lstm_size, dir):
    file_name = dir + 'out_lstm_size_' + str(lstm_size) + '.txt'
    data = []
    validation_data = []
    X = []
    Y = []
    Y1 = []
    with open(file_name, encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            if(cnt >= start and (cnt - start) % one_epoch >= one_epoch - mean_size):
                data.append(float(line.split()[offset].strip('.')))
                validation_data.append(float(line.split()[offset+2].split('...')[0]))
            cnt = cnt + 1
    for i in range(0, len(data), mean_size):
        X.append(i // mean_size + 1)
        Y.append(np.mean(data[i: i+mean_size]))
        Y1.append(np.mean(validation_data[i: i+mean_size]))
    return X, Y, Y1

X = []
Y = []
plt.ylabel('Train Loss')
plt.xlabel('Layer Number')
for lstm_size in lstm_sizes:
    x, y, _ = get_X_Y(lstm_size, dir)
    plt.plot(x, y, label='lstm-size='+str(lstm_size))
plt.legend(loc='upper left')
plt.show()
