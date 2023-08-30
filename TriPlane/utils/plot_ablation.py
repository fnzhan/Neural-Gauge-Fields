import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(25.6, 4.8))
plt.rcParams["mathtext.fontset"] = "cm"

# hosrse.
Original =  [25.28, 25.28, 25.28, 25.28, 25.28]
StruReg =  [31.28, 31.28, 31.28, 31.28, 31.28]
Cycle =     [26.38, 30.34, 31.59, 31.63, 31.67]
InfoReg =   [26.67, 30.94, 32.57, 32.52, 32.61]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [0.01,   0.1,   1,  10,  100]
plt.subplot(1, 4, 1)
plt.xscale('log')
plt.grid(visible=True)
plt.xticks(x)
plt.plot(x, InfoReg, marker='o', label='InfoReg', color='limegreen')
plt.plot(x, Cycle, marker='*', markersize=8, label='Cycle Loss')
plt.plot(x, StruReg, marker='v', markersize=8, label='StruReg')
plt.plot(x, Original, marker='v', markersize=8, label='Original', color='slateblue')

plt.legend(loc='center right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Regularization Weight $\\gamma$", fontsize=14)
plt.ylabel("PSNR", fontsize=12)
plt.title("3D Space $\\rightarrow$ 2D Plane", fontsize=16)
# plt.savefig('unpaired_graph.png', bbox_inches='tight')


Original =  [24.79, 24.79, 24.79, 24.79, 24.79]
Cycle =     [24.83, 24.94, 24.88, 25.02, 24.95]
InfoReg =   [26.23, 26.94, 27.37, 27.45, 27.44]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [0.01,   0.1,   1,  10,  100]
plt.subplot(1, 4, 2)
plt.xscale('log')
plt.grid(visible=True)
plt.xticks(x)
plt.plot(x, InfoReg, marker='o', label='InfoReg', color='limegreen')
plt.plot(x, Cycle, marker='*', markersize=8, label='Cycle Loss')
plt.plot(x, Original, marker='v', markersize=8, label='Original', color='slateblue')
plt.legend(loc='center right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Regularization Weight $\\gamma$", fontsize=14)
plt.ylabel("PSNR", fontsize=12)
plt.title("3D Space $\\rightarrow$ 256 Vectors", fontsize=16)
# plt.savefig('unpaired_graph.png', bbox_inches='tight')




Original =  [25.28, 25.28, 25.28, 25.28, 25.28]
StruReg =  [31.28, 31.28, 31.28, 31.28, 31.28]
# Cycle =     [29.58, 31.34, 32.09, 31.49, 31.17]
InfoReg =   [29.83, 31.74, 32.57, 32.64, 32.01]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [0.01,   0.1,   1,  10,  100]
plt.subplot(1, 4, 3)
plt.xscale('log')
plt.grid(visible=True)
plt.xticks(x)
plt.plot(x, InfoReg, marker='o', label='InfoReg', color='limegreen')
plt.plot(x, StruReg, marker='v', markersize=8, label='StruReg', color='darkorange')
# plt.plot(x, Original, marker='v', markersize=8, label='Original', color='slateblue')
# plt.plot(x, Cycle, marker='*', markersize=8, label='Cycle')

plt.legend(loc='lower right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Prior Discrepancy Weight $\\epsilon$", fontsize=14)
plt.ylabel("PSNR", fontsize=12)
plt.title("3D Space $\\rightarrow$ 2D Plane", fontsize=16)



Original =  [24.79, 24.79, 24.79, 24.79, 24.79]
# Cycle =     [24.83, 25.14, 25.11, 24.64, 24.37]
InfoReg =   [25.33, 26.74, 27.37, 27.21, 26.33]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [0.01,   0.1,   1,  10,  100]
plt.subplot(1, 4, 4)
plt.xscale('log')
plt.grid(visible=True)
plt.xticks(x)
plt.plot(x, InfoReg, marker='o', label='InfoReg', color='limegreen')
plt.plot(x, Original, marker='v', markersize=8, label='Original', color='slateblue')
# plt.plot(x, Cycle, marker='*', markersize=8, label='Cycle')

plt.legend(loc='center right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Prior Discrepancy Weight $\\epsilon$", fontsize=14)
plt.ylabel("PSNR", fontsize=12)
plt.title("3D Space $\\rightarrow$ 256 Vectors", fontsize=16)
plt.savefig('im_ablation.pdf', bbox_inches='tight')
# plt.show()
