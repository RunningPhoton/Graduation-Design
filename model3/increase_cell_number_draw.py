import matplotlib.pyplot as plt
import numpy as np


lstm_sizes = [32, 64, 128, 192, 256, 320, 448, 600]
offset = 9
X = []
Y = []
Y1 = []
test_data = []
start = 2
st_read = 66
ed_read = 70
dir = '01/'
for lstm_size in lstm_sizes:
    X.append(lstm_size)
    file_name = 'out_lstm_size_' + str(lstm_size) + '.txt'

    with open(dir + file_name, encoding='UTF-8') as f:
        data = []
        validation_data = []
        all = f.readlines()
        test_data.append(float(all[ed_read].split()[6]))
        for _ in range(st_read, ed_read):
            line = all[_]
            print(_, line)
            data.append(float(line.split()[offset]))
            validation_data.append(float(line.split()[offset+8]))
        Y.append(np.mean(data))
        Y1.append(np.mean(validation_data))

plt.ylabel('Accuracy')
plt.xlabel('Layer Number')
plt.plot(X, Y, color='blue', label='train_accuracy')
plt.plot(X, Y1, color='green', label='validation_accuracy')
plt.plot(X, test_data, color='red', label='test_accuracy')
plt.legend(loc='lower right')
plt.show()
