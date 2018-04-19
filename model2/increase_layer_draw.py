import matplotlib.pyplot as plt
import numpy as np

file_name = 'out_lstm_size_192.txt'
dir = '01/'
start = 3
one_epoch = 102
mean_size = 10
offset = 7
data = []
validation_data = []
X = []
Y = []
Y1 = []
with open(dir + file_name, encoding='UTF-8') as f:
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

plt.ylabel('Loss')
plt.xlabel('Layer Number')
plt.plot(X, Y, color='blue', label='train_loss')
plt.plot(X, Y1, color='red', label='validation_loss')
plt.legend(loc='upper left')
plt.show()
