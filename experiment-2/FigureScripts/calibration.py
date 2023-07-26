import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def create_calibration():

    def funlin(x, a, b):
        return a*x + b

    Am241_xs = np.array([900.257, 1091.514, 1145, 1340.27])
    Am241_peaks = np.array([13.9, 16.84, 17.7502, 20.7848])

    Fe55_xs = np.array([382.27, 420.12])
    Fe55_peaks = np.array([(6.404+6.391)/2, 7.058])

    xs = np.concatenate((Fe55_xs, Am241_xs))
    ys = np.concatenate((Fe55_peaks, Am241_peaks))

    popt, _ = curve_fit(funlin, xs, ys)

    plt.errorbar(xs, ys, xerr=1, fmt='o', barsabove=1, label="Data")
    label = rf"$E(n)={popt[1]:.4f}\,\,\mathrm{{keV}}+{popt[0]:.4f}" + r"\frac{\mathrm{keV}}{\mathrm{channel}}\cdot n$"
    print(label)
    plt.plot(np.linspace(0, max(xs)), funlin(np.linspace(0, max(xs)), *popt), label=label)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
            }

    plt.text(400, 10, "Mangler usikkerheder!", fontdict=font)

    plt.title("Energy Calibration")
    plt.xlabel("Channel")
    plt.ylabel("Energy [keV]")
    plt.legend()
    plt.grid()

    plt.savefig("Figures/calibration.pdf")