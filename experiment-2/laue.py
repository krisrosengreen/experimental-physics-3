import helpers.pathfinder as pf
import helpers.mcaloader as mca
from helpers.data import DataStructure as DS
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import analyzepeaks as ap

def funlin(x, a, b):
    return a*x+b

if __name__ == "__main__":
    dat = DS()
    dat.loadfile("laue.txt")

    analyzer = ap.AnalyzeEX2({})
    xscale = analyzer.xscale

    xtops, ytops, xtops_error = dat.df.to_numpy().T
    xtops = xtops[4:]
    xtops_error = xtops_error[4:]
    ns = np.arange(5, 9)

    popt, pcov = curve_fit(funlin, xtops, ns, sigma=xtops_error, absolute_sigma=True)

    print(xtops_error)

    errors = np.sqrt(np.diag(pcov))


    ls = np.linspace(xtops[0], xtops[-1])
    ys = funlin(ls, *popt)

    plt.plot(ls, ys, c="m", label=rf"$n(E)=({popt[0]:.4f}\pm{errors[0]:.4f}\frac{{1}}{{\mathrm{{keV}}}})\cdot E {popt[1]:4f}$")
    plt.xlabel("Energy $E$ [keV]")
    plt.ylabel("Peak number $n$")
    plt.title("Peak number fitted to energy")

    plt.errorbar(xtops, ns, label="Data", xerr=xtops_error, fmt='o', capsize=4, ms=2)
    plt.legend()
    plt.grid()
    plt.savefig("Figures/diffraction.pdf")
    #plt.show()
