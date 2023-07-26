# Allows use of files in helpers
# Lines of code that allow use of files in helpers module
import os, sys
from re import I
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# Useful modules
import helpers.mcaloader as mca
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import analyzepeaks as ap

def gauss(x, H, A, x0, sigma):
    """
    Gaussian function used for fitting peaks.
    """
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

data = mca.quick_load_mca("Am241w2")
xs = np.arange(0, len(data))

res = find_peaks(data, prominence=5, distance=20,
                    height=max(data) / 10)[0][0]

xscale = ap.AnalyzeEX2({}).xscale

xs_cut = xs[res-20:res+20]
data_cut = data[res-20:res+20]

popt,pcov = curve_fit(gauss, xs_cut, data_cut, p0=[0,300,900,10])

ls = np.linspace(min(xs_cut), max(xs_cut))

plt.plot(ls*xscale, gauss(ls, *popt), label="Best fit")

plt.errorbar(xs_cut*xscale, data_cut, yerr=np.sqrt(data_cut), fmt='o', capsize=3, label="Data")
plt.title("Example of Gaussian function fitted to a peak")
plt.xlabel("Energyy [keV]")
plt.ylabel("Counts")
plt.grid()
plt.legend()
plt.savefig("Figures/gauss-example.pdf")