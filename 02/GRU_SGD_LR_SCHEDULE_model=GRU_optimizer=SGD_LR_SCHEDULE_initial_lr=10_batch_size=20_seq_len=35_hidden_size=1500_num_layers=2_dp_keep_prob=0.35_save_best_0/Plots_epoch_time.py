#import matplotlib
## Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')


import numpy as np
import matplotlib.pyplot as plt

timeline=[]
num_lines = 0
with open("log.txt", 'r') as f:
    for line in f:
        token = line.split()
        timeline.append(float(token[-1]))
        num_lines += 1
wall_clock_time = np.cumsum(timeline)

x = np.load("learning_curves.npy")[()]
train_ppls = x['train_ppls']
val_ppls = x['val_ppls']
train_losses = x['train_losses']
val_losses = x['val_losses']
epoch_num = len(train_ppls)

fig1, fig2= plt.subplots()
plt.plot(range(1,1+epoch_num), train_ppls, label="Train")
plt.plot(range(1,1+epoch_num), val_ppls, label="Validation", color='red', markersize=12)
plt.legend(['Training', 'Validation'], loc='upper right')
plt.xlabel("Epoch")
plt.ylabel("PPL")
plt.savefig('PPL_over_ecpochs.png', format='png', dpi=150)
plt.show()

plt.plot(wall_clock_time, train_ppls, label="Train")
plt.plot(wall_clock_time, val_ppls, label="Validation", color='red')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.xlabel("Time")
plt.ylabel("PPL")
plt.savefig('PPL_over_time.png', format='png', dpi=150)
plt.show()


