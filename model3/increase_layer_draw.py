import matplotlib.pyplot as plt
import numpy as np

file_name = 'out_lstm_size_448.txt'
dir = '01/'
start = 2
one_epoch = 23
mean_size = 3
offset = 9
data = []
validation_data = []
test_data = []
X = []
Y = []
Y1 = []
with open(dir + file_name, encoding='UTF-8') as f:
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
print(data)
print(len(validation_data))
print(len(test_data))

for i in range(0, len(data), mean_size - 1):
    X.append(i // mean_size)
    Y.append(np.mean(data[i: i+mean_size]))
    Y1.append(np.mean(validation_data[i: i+mean_size]))

plt.ylabel('Accuracy')
plt.xlabel('Layer Number')
plt.plot(X, Y, color='blue', label='train_accuracy')
plt.plot(X, Y1, color='green', label='validation_accuracy')
plt.plot(X, test_data, color='red', label='test_accuracy')
plt.legend(loc='upper right')
plt.show()
