from helpers.mcaloader import load_mca as mca
import helpers.pathfinder as pf
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import analyzepeaks as ap
from scipy.optimize import curve_fit
import analyzepeaks as ap

if __name__ == "__main__":
    files = pf.dir_file_crawler("data/")

    xscale = ap.AnalyzeEX2({}).xscale

    with45 = pf.getfile_in_L("36KVGlimmer45degw3", files)
    without45 = pf.getfile_in_L("36KVGlimmern45degw3", files)

    with45_ys = mca(with45)
    without45_ys = mca(without45)
    xs = np.arange(0, len(with45_ys))*xscale

    #plt.errorbar(xs, with45_ys, yerr=np.sqrt(with45_ys), fmt='o', capsize=2, ms=1, elinewidth=1)
    #plt.errorbar(xs, without45_ys, yerr=np.sqrt(without45_ys), fmt='o', capsize=2, capthick=1, ms=1, elinewidth=1)

    plt.plot(xs, with45_ys, label=r"Data on glimmer $45^\circ$")
    plt.plot(xs, without45_ys, label=r"Data on glimmer not $45^\circ$")

    plt.fill_between(np.append(xs, xs[::-1]), np.append(with45_ys+np.sqrt(with45_ys), with45_ys[::-1]-np.sqrt(with45_ys)[::-1]), facecolor=(0,0,1,0.2), label=r"Uncertainty on glimmer $45^\circ$")
    plt.fill_between(np.append(xs, xs[::-1]), np.append(without45_ys+np.sqrt(without45_ys), without45_ys[::-1]-np.sqrt(without45_ys)[::-1]), facecolor=(1,0,0,0.2), label=r"Uncertainty on glimmer not $45^\circ$")

    plt.xlim(5,12)
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")
    plt.title(r"Plot of crystal positioned at $45^\circ$ and not $45^\circ$")
    plt.legend()
    plt.grid()
    # plt.savefig("Figures/glimmer_raw.pdf")
    plt.show()