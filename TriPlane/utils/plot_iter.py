# from numpy.random import RandomState
import matplotlib.pyplot as plt
# from matplotlib import ticker

# plt.figure(figsize=(25.6, 4.8))
plt.rcParams["mathtext.fontset"] = "cm"

# plt.grid(visible=True)
# ax = plt.gca()

# f = open('log_pe.txt', "r")
txt_dir = '/CT/Neural-Gauge-Fields/work/ICLR-Neural-Gauge-Fields/log/TriplaneNGF/'
idx = 0
iter, psnr = [], []
with open(txt_dir + "log_256_gaugetri_4096.txt",'r') as data_file:
    for line in data_file:
        data = line.split()
        iter.append(int(data[1][:5]))
        psnr.append(float(data[4]))
        # iter, psnr = int(data[1][:5]), float(data[4])
        idx += 1
        print(idx)
        if idx > 1400:
            break
idx = 0
iter2, psnr2 = [], []
with open(txt_dir + "log_256_bs4096.txt",'r') as data_file:
    for line in data_file:
        data = line.split()
        # print(data)
        iter2.append(int(data[1][:5]))
        psnr2.append(float(data[4]))
        # iter, psnr = int(data[1][:5]), float(data[4]).
        idx += 1
        # print(int(data[1][:5]))
        if idx > 1400:
            break


plt.plot(iter2, psnr2, label='Triplane', color='red', linewidth=1)
plt.plot(iter, psnr, label='Triplane + Gauge Fields', color='limegreen', linewidth=1)
plt.legend(loc='center right', fontsize=12.5)

plt.savefig('infonest.png')


