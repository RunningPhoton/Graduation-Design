import matplotlib.pyplot as plt
import numpy as np

file_name = 'out_lstm_size_384.txt'
dir = '01/'
start = 5
one_epoch = 80
mean_size = 5
offset = 5
data = []
X = []
Y = []
with open(dir + file_name, encoding='UTF-8') as f:
    cnt = 0
    for line in f:
        if(cnt >= start and (cnt - start) % one_epoch >= one_epoch - mean_size):
            data.append(float(line.split()[offset].strip('.')))
            # print(cnt - start, line.split()[offset].strip('.'))
        cnt = cnt + 1
print(data)
for i in range(0, len(data), mean_size):
    X.append(i // mean_size + 1)
    Y.append(np.mean(data[i:i + mean_size]))

plt.ylabel('Loss')
plt.xlabel('Layer Number')
plt.plot(X, Y)
plt.show()
