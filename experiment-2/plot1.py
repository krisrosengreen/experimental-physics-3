from helpers.mcaloader import load_mca as mca
import helpers.pathfinder as pf
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import analyzepeaks as ap
from scipy.optimize import curve_fit
import analyzepeaks as ap
from math import isclose


def line(x, a):
    return a

def closest(input_list, input_value):
    arr = np.asarray(input_list)
    i = (np.abs(arr - input_value)).argmin()
    return arr[i], i

if __name__ == "__main__":
    files = pf.dir_file_crawler("data/")

    xscale = ap.AnalyzeEX2({}).xscale

    I0_path = pf.getfile_in_L("36KV10minw3", files)
    I1 = "data/Dataw3/36KVabs1w3igen10min.mca"
    I2 = "data/Dataw3/36KVabs2w3igen10min.mca"
    I3 = "data/Dataw3/36KVabs3w3igen10min.mca"

    data_I0 = mca(I0_path)
    I0_err = np.sqrt(data_I0)
    data_I1 = mca(I1)
    I1_err = np.sqrt(data_I1)
    data_I2 = mca(I2)
    I2_err = np.sqrt(data_I2)
    data_I3 = mca(I3)
    I3_err = np.sqrt(data_I3)

    xscale = ap.AnalyzeEX2({}).xscale#+7e-4
    xs = np.arange(0, len(data_I0))*xscale

    plt.plot(xs, data_I0, label=r'$I_0$', color='tab:blue')
    plt.plot(xs, data_I1, label='Absorber 1', color='tab:orange')
    plt.plot(xs, data_I3, label='Absorber 3', color='tab:red')
    plt.plot(xs, data_I2, label='Absorber 2', color='tab:green')
    plt.xlim(5,30)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Energy [keV]')
    plt.ylabel('Counts')
    plt.legend()
    plt.title('Absorber Spectra')
    plt.tight_layout()
    # Hardcode fy fy
    plt.savefig('Figures\\absorber_spectra.pdf')
    plt.show()

    plt.cla()

    I0_I1 = data_I0 / (data_I1 + 1e-20)
    I0_I2 = data_I0 / (data_I2 + 1e-20)
    I0_I3 = data_I0 / (data_I3 + 1e-20)
    Es1,mus1 = np.genfromtxt("data/linear_attenuation_mo.csv", delimiter='').T
    Es2,mus2 = np.genfromtxt('data/linear_attenuation_Ag.txt',delimiter='').T
    Es3,mus3 = np.genfromtxt('data/linear_attenuation_Au.txt',delimiter='').T
    new_mus1 = np.array([])
    new_mus2 = np.array([])
    new_mus3 = np.array([])
    _, i = closest(xs,30)
    _, j = closest(xs,5)
    xs = xs[j:i]
    I0_I1 = I0_I1[j:i]
    I0_I2 = I0_I2[j:i]
    I0_I3 = I0_I3[j:i]

    for x in xs:
        val1, i1 = closest(Es1, x)
        val2, i2 = closest(Es2, x)
        val3, i3 = closest(Es3, x)
        new_mus1 = np.append(new_mus1, mus1[i1])
        new_mus2 = np.append(new_mus2, mus2[i2])
        new_mus3 = np.append(new_mus3, mus3[i3])

    t1 = np.log(I0_I1) / new_mus1
    t2 = np.log(I0_I2) / new_mus2
    t3 = np.log(I0_I3) / new_mus3
    mt1 = np.mean(t1[np.where(t1 > -1e50)]).round(6)
    mt2 = np.mean(t2[np.where(t2 > -1e50)]).round(6)
    mt3 = np.mean(t3[np.where(t3 > -1e50)]).round(6)
    s1 = np.std(t1[np.where(t1 > -1e50)]).round(6)
    s2 = np.std(t2[np.where(t2 > -1e50)]).round(6)
    s3 = np.std(t3[np.where(t3 > -1e50)]).round(6)
    print(mt1,mt2,mt3)
    plt.errorbar(xs, t1, xerr=0.5*xscale, color='tab:orange', fmt='.',label='Absorber 1')
    plt.errorbar(xs, t2, xerr=0.5*xscale, color='tab:green', fmt='.', label='Absorber 2')
    plt.errorbar(xs, t3, xerr=0.5*xscale, color='tab:red', fmt='.', label='Absorber 3')
    plt.xlabel('E [keV]')
    plt.ylabel(r'$x=-\frac{\log(I_0/I)}{\mu(E)}$ [cm]')
    plt.grid()
    #plt.xlim(0,36)
    #plt.ylim(-0.01,0.025)
    plt.title('Absorber thickness')
    plt.tight_layout()

    test_error1 = np.sqrt(1/np.multiply(np.sqrt(data_I0[j:i]), new_mus1) + 1/np.multiply(np.sqrt(data_I1[j:i]), new_mus1))
    test_error2 = np.sqrt(1/np.multiply(np.sqrt(data_I0[j:i]), new_mus2) + 1/np.multiply(np.sqrt(data_I2[j:i]), new_mus2))
    test_error3 = np.sqrt(1/np.multiply(np.sqrt(data_I0[j:i]), new_mus3) + 1/np.multiply(np.sqrt(data_I3[j:i]), new_mus3))

    popt1,pcov1 = curve_fit(line, xs, t1[np.where(t1 > -1e50)], absolute_sigma=True, sigma=test_error1)
    popt2,pcov2 = curve_fit(line, xs, t2[np.where(t2 > -1e50)], absolute_sigma=True, sigma=test_error2)
    popt3,pcov3 = curve_fit(line, xs, t3[np.where(t3 > -1e50)], absolute_sigma=True, sigma=test_error3)

    error1 = np.sqrt(np.diag(pcov1)).round(6)
    error2 = np.sqrt(np.diag(pcov2)).round(6)
    error3 = np.sqrt(np.diag(pcov3)).round(6)

    print("vals:", popt1[0], popt2[0], popt3[0])
    print("errs", error1[0], error2[0], error3[0])

    xerr = 0.5*xscale.round(6)
    plt.text(4,0.035,r'$\bar{{x}}_1={m1}\pm{p}$'.format(m1=mt1))
    plt.text(4,0.032,r'$\bar{{x}}_2={m2}$'.format(m2=mt2))
    plt.text(4,0.029,r'$\bar{{x}}_3={m3}$'.format(m3=mt3))
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('Figures\\Absorber_thickness.pdf')
    plt.show()
