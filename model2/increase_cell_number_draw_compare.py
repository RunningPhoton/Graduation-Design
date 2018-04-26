import matplotlib.pyplot as plt
import numpy as np


lstm_sizes = [32, 64, 96, 128, 160, 192]
offset = 7
start = 3
st_read = 400
ed_read = 410
def get_X_Y_Y1(dir, f_name):
    X = []
    Y = []
    Y1 = []
    for lstm_size in lstm_sizes:
        X.append(lstm_size)
        file_name = f_name + str(lstm_size) + '.txt'

        with open(dir + file_name, encoding='UTF-8') as f:
            data = []
            validation_data = []
            all = f.readlines()
            for _ in range(st_read, ed_read):
                line = all[_]
                print(_, line)
                data.append(float(line.split()[offset]))
                validation_data.append(float(line.split()[offset+4]))
            Y.append(np.mean(data))
            Y1.append(np.mean(validation_data))
    return X, Y, Y1

x, a, b = get_X_Y_Y1('01/', 'out_lstm_size_')
_, a1, b1 = get_X_Y_Y1('02/', 'out2_lstm_size_')
plt.ylabel('Loss')
plt.xlabel('Cell Number')
plt.plot(x, a, color='blue', label='simple_loss')
plt.plot(x, a1, color='red', label='complex_loss')
plt.legend(loc='upper right')
plt.show()
