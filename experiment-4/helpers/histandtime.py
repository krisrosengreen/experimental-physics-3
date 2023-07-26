import helpers.pathfinder as pf
import numpy as np
import pandas as pd

def get_hist_and_time(filename, low_lim = 70, up_lim = 2000):
    L = pf.dir_file_crawler('Data/')
    file_path = pf.getfile_in_L(filename, L)
    data = pd.read_csv(file_path, skiprows = np.arange(0,5), delimiter=' ')
    print(file_path)

    data = data.to_numpy()
    time_interval= data[-1][0]-data[0][0]
    ch_co = data.transpose()[1]
    ch_co_cut = [i for i in ch_co if i>low_lim and  i< up_lim]
    co, ch = np.histogram(ch_co_cut, bins = np.arange(low_lim, up_lim))
    ch = ch[:-1]
    
    return ch, co, time_interval
