import matplotlib.pyplot as plt
import numpy as np

# lstm_sizes = [32, 64, 128, 192, 256, 320, 448, 600]

dir = '01/'
start = 2
one_epoch = 23
mean_size = 3
offset = 9
def get_X_Y(dir, lstm_size):
    file_name = dir + 'out_lstm_size_' + str(lstm_size) + '.txt'
    data = []
    validation_data = []
    test_data = []
    X = []
    Y = []
    Y1 = []
    with open(file_name, encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            # if(cnt >= start):
            #     print(cnt - start, line)
            if(cnt >= start and (cnt - start) % one_epoch >= one_epoch - mean_size):
                # print(cnt - start, line)
                if((cnt - start) % one_epoch == one_epoch - 1):
                    test_data.append(float(line.split()[6]))
                else:
                    data.append(float(line.split()[offset]))
                    validation_data.append(float(line.split()[offset+8]))
            cnt = cnt + 1
    for i in range(0, len(data), mean_size - 1):
        X.append(i // mean_size)
        Y.append(np.mean(data[i: i+mean_size]))
        Y1.append(np.mean(validation_data[i: i+mean_size]))
    return X, Y, Y1
plt.ylabel('Accuracy')
plt.xlabel('Layer Number')
for lstm_size in [32, 64, 128]:
    X, Y, _ = get_X_Y(dir, lstm_size)
    plt.plot(X, Y, label='lstm-size='+str(lstm_size))
plt.legend(loc='lower left')
plt.show()
