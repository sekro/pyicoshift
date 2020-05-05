from pyicoshift import Icoshift
from scipy.io import loadmat
import matplotlib.pyplot as plt

wine = loadmat('../data/WineData.mat')


X = wine['X']

ppm = wine['ppm'][0]

wine_ints = wine['wine_ints'][0]
custom_intervals = []
for i in range(0, wine_ints.shape[0], 2):
    custom_intervals.append((int(wine_ints[i]),int(wine_ints[i+1])))
print(custom_intervals)

# instance icoshift obj
reference = Icoshift()
fix_n_int = Icoshift()
fix_int_size = Icoshift()
user_int1 = Icoshift()
user_int2 = Icoshift()
user_int3 = Icoshift()

# set model/dataset signal_names
reference.name = 'Wine 1D NMR - referenced to Lac'
fix_n_int.name = 'Wine 1D NMR - 50 intervals'
fix_int_size.name = 'Wine 1D NMR - intervals width 800 pts'
user_int1.name = 'Wine 1D NMR - user defined intervals'
user_int2.name = 'Wine 1D NMR - user defined intervals'
user_int3.name = 'Wine 1D NMR - user defined intervals'

# optional set log lvl
reference.loglvl = 'debug'

# assign signals
reference.signals = X
fix_n_int.signals = X
fix_int_size.signals = X
user_int1.signals = X
user_int2.signals = X
user_int3.signals = X

# assign scales for figures
reference.unit_vector = ppm
fix_n_int.unit_vector = ppm
fix_int_size.unit_vector = ppm
user_int1.unit_vector = ppm
user_int2.unit_vector = ppm
user_int3.unit_vector = ppm

# set intervals / align mode
# shift relative to lac
reference.inter = ('shift_relative_to_region',(7551,7751))

fix_n_int.inter = ('n_intervals', 50)

fix_int_size.inter = ('fixed_interval_length', 800)
user_int1.inter = custom_intervals
print(user_int1.inter)
user_int2.inter = custom_intervals
user_int3.inter = custom_intervals

# non default configs
fix_n_int.target = 'maxcorr'
fix_int_size.max_shift = 'best'
user_int2.global_pre_align = True
user_int2.target = 'average2'
user_int3.global_pre_align = True
user_int3.target = 'max'
user_int3.fill_mode = 'nan'

# run the shifting
print('---reference---')
reference.run()
print('---fix n int---')
fix_n_int.run()
print(fix_n_int.inter)
print('---fix in size---')
fix_int_size.run()
print('---user 1---')
user_int1.run()
print('---user 2---')
user_int2.run()
print('---user 3---')
user_int3.run()


#make figures and plot
reference.make_figures()
fix_int_size.make_figures()
fix_n_int.make_figures()
user_int1.make_figures()
user_int2.make_figures()
user_int3.make_figures()

plt.show()
