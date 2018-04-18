import matplotlib.pyplot as plt
import numpy as np


lstm_sizes = [128, 192, 256, 384, 512]
offset = 5
X = []
Y = []
start = 5
mean_size = 5
st_read = 238
ed_read = 245
dir = '01/'
for lstm_size in lstm_sizes:
    X.append(lstm_size)
    file_name = 'out_lstm_size_' + str(lstm_size) + '.txt'

    with open(dir + file_name) as f:
        data = []
        all = f.readlines()
        for _ in range(st_read, ed_read):
            line = all[_]
            print(_, line)
            data.append(float(line.split()[offset].strip('.')))
        Y.append(np.mean(data))

plt.ylabel('Loss')
plt.xlabel('Layer Number')
plt.plot(X, Y)
plt.show()