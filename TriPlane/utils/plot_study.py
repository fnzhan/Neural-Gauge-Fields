import numpy as np
import matplotlib.pyplot as plt


plt.figure(figsize=(25.6, 4.8))
plt.rcParams["mathtext.fontset"] = "cm"



Original =  [23.99, 28.14, 29.97, 31.76, 33.89, 34.12]
InfoReg =  [27.76, 30.82, 31.96, 32.66, 34.08, 34.20]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [1,    4,     8,  16,  32,  64]
plt.subplot(1, 4, 1)
# plt.xscale('log')
plt.grid(visible=True)
plt.xticks(x)
plt.plot(x, InfoReg, marker='o', label='Original+Reg', color='limegreen')
plt.plot(x, Original, marker='v', markersize=8, label='Original', color='slateblue')
plt.legend(loc='center right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Top-K Vectors", fontsize=14)
plt.ylabel("PSNR", fontsize=12)
plt.title("Rendering Performance (256 Vectors)", fontsize=16)
# plt.savefig('unpaired_graph.png', bbox_inches='tight')




# hosrse.
Original =  [0.12, 0.39, 0.91, 1.0, 1.0]
# Cycle =     [29.58, 31.34, 32.09, 31.49, 31.17]
# InfoReg =  [1.0, 1.0, 1.0, 1.0, 1.0]
InfoReg =  [0.08, 0.25, 0.37, 0.74, 1.0]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [1,   4,  8,  16, 32]
plt.subplot(1, 4, 2)
# plt.xscale('log')
plt.grid(visible=True)
plt.xticks(x)
plt.plot(x, Original, marker='v', markersize=8, label='256 Vectors', color='dodgerblue')
plt.plot(x, InfoReg, marker='o', label='1024 Vectos', color='orangered')
# plt.plot(x, Cycle, marker='*', markersize=8, label='Original')

plt.legend(loc='center right', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Top-K Vectors", fontsize=14)
plt.ylabel("Ratio", fontsize=12)
plt.title("Codebook Utilization", fontsize=16)
# plt.savefig('unpaired_graph.png', bbox_inches='tight')








# Original =  [12, 22, 33, 41, 62]
# Cycle =     [29.58, 31.34, 32.09, 31.49, 31.17]
# InfoReg =  [0.159, 0.167, 0.175, 0.186, 0.211, 0.261, 0.365, 0.620]
InfoReg =  [0.099, 0.107, 0.115, 0.126, 0.151, 0.191, 0.275, 0.385]
SHF =  [0.096, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [1,   4,  8,  16, 32, 64, 128, 256]
plt.subplot(1, 4, 3)
plt.xscale('log')
plt.grid(visible=True)
plt.xticks(x)
# plt.plot(x, Original, marker='v', markersize=8, label='StruReg')
plt.plot(x, InfoReg, marker='*', markersize=8, label='Learned Mapping', color='darkorange')
plt.plot(x, SHF, marker='v', markersize=8, label='Spatial Hash', color='slateblue')
# plt.plot(x, InfoReg, marker='o', label='InfoReg')
plt.legend(loc='center left', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Top-K Vectors", fontsize=14)
plt.ylabel("Second", fontsize=12)
plt.title("Computation Cost (256 Vectors)", fontsize=16)



# InfoReg =  [0.178, 0.196, 0.219, 0.267, 0.360, 0.552, 0.926, 1.62]
InfoReg =  [0.164, 0.171, 0.183, 0.204, 0.242, 0.310, 0.382, 0.492, 0.606, 0.873]
SHF =  [0.096, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096, 0.096]
# easy_y = [52.7, 48.61, 46.92, 46.12, 45.51]
x =         [1,   4,  8,  16, 32, 64, 128, 256, 512, 1024]
plt.subplot(1, 4, 4)
plt.xscale('log')
plt.grid(visible=True)
plt.xticks(x)
plt.plot(x, InfoReg, marker='*', markersize=8, label='Learned Mapping', color='darkorange')
plt.plot(x, SHF, marker='v', markersize=8, label='Spatial Hash', color='slateblue')
# plt.plot(x, InfoReg, marker='o', label='InfoReg')
plt.legend(loc='center left', fontsize='xx-large')
plt.tick_params(axis='both', which='major', labelsize=12)
plt.xlabel("Top-K Vectors", fontsize=14)
plt.ylabel("Second", fontsize=12)
plt.title("Computation Cost (1024 Vectors)", fontsize=16)

plt.savefig('im_study.pdf', bbox_inches='tight')
# plt.show()
