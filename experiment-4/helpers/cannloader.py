import pandas as pd
import numpy as np
import collections as cs
from helpers.pathfinder import isfile_in_L
import helpers.pathfinder as pf
import os



def load_file(filename, lowest=30, highest=2000) -> list:
    hists_folder = "Data/Hists/"
    L = pf.dir_file_crawler(hists_folder)

    name = filename.split("/")[-1].split('.')[0]+"_hist"

    if pf.isfile_in_L(name, L):
        print("Loading saved hist...")
        xs, ys = np.genfromtxt(pf.getfile_in_L(name, L), delimiter=",", skip_header=1).T
        return xs[lowest:highest], ys[lowest:highest], None

    df = pd.read_csv(filename, skiprows=np.arange(0,5), delimiter=' ')
    res = df.to_numpy()
    column_counts = res[:,1]
    column_time = res[:, 0]
    a = cs.Counter(column_counts)
    a = cs.OrderedDict(sorted(a.items()))

    # Discard the first 30 rows!

    xs = np.fromiter(a.keys(), dtype=float)
    ys = np.fromiter(a.values(), dtype=float)

    xs = xs[lowest:highest]
    ys = ys[lowest:highest]

    pd.DataFrame.from_dict({"xs": xs, "ys": ys}).to_csv(hists_folder + name + ".txt", index=False)

    return xs, ys, column_time

def load_file_raw(filename) -> np.array:
    df = pd.read_csv(filename, skiprows=np.arange(0,5), delimiter=' ')
    return df.to_numpy()
