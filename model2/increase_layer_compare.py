import matplotlib.pyplot as plt
import numpy as np


lstm_sizes = [32, 64, 96, 128, 160, 192]
dir = '01/'
start = 3
one_epoch = 102
mean_size = 10
offset = 7
def get_X_Y(dir, lstm_size):
    file_name = dir + 'out_lstm_size_' + str(lstm_size) + '.txt'
    data = []
    validation_data = []
    X = []
    Y = []
    Y1 = []
    with open(file_name, encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            # if(cnt >= start):
            #     print(cnt - start, line)
            if(cnt >= start and (cnt - start) % one_epoch >= one_epoch - mean_size):
                data.append(float(line.split()[offset]))
                validation_data.append(float(line.split()[offset+4]))
                # print(cnt - start, line)
            cnt = cnt + 1
    # print(len(data))
    for i in range(0, len(data), mean_size):
        X.append(i // mean_size + 1)
        Y.append(np.mean(data[i: i+mean_size]))
        Y1.append(np.mean(validation_data[i: i+mean_size]))
    return X, Y, Y1

plt.ylabel('Loss')
plt.xlabel('Layer Number')
for lstm_size in lstm_sizes:
    X, Y, _ = get_X_Y(dir, lstm_size)
    plt.plot(X, Y, label='lstm-size='+str(lstm_size))
plt.legend(loc='upper left')
plt.show()
