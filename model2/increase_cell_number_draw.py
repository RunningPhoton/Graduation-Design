import matplotlib.pyplot as plt
import numpy as np


lstm_sizes = [32, 64, 96, 128, 160, 192]
offset = 7
X = []
Y = []
Y1 = []
start = 3
st_read = 300
ed_read = 309
dir = '01/'
for lstm_size in lstm_sizes:
    X.append(lstm_size)
    file_name = 'out_lstm_size_' + str(lstm_size) + '.txt'

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

plt.ylabel('Loss')
plt.xlabel('Layer Number')
plt.plot(X, Y, color='blue', label='train_loss')
plt.plot(X, Y1, color='red', label='validation_loss')
plt.legend(loc='upper right')
plt.show()
