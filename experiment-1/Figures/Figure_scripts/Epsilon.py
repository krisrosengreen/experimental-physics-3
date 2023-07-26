"""
@Author: Andreas
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# N_forventet = Aktivitet * Dt * abundance/100
# Aktivitet = 1.08muCi -> Bq
# epsilon(E) = N_målt/N_forventet ~ a*E^b
plt.rc('font',size=14)

activity = 4.8 * 1e-6 * 37000000000  # Bq


Dt = (22444925848 - 3127)*1e-8  # s

eV_ch = 0.740252965505167  # eV

gauss = lambda x, a, b, c, d: a * np.exp(-((b - x) ** 2 / (2 * c ** 2))) + d  # The fitting function


def get_data(data: list):
    """
    Get data as an array of [channel, counts] in stead of [time, channel]
    :param data: array of data [time, channel]
    :return:
        channels: numpy.array(channels)
        counts: numpy.array(counts)
    """
    # The channels must the numbers from min(channels) to max(channels) (not all channels appear in the data
    channels = np.arange(int(min(data[1])), int(max(data[1]) + 1), 1)
    counts = np.zeros([channels.size])

    # Counts are the number of times one channel appears in the data
    for c in data[1]:
        counts[int(c)] += 1
    return channels * eV_ch, counts


# Getting the data file
file = np.loadtxt(
    '../../Data/Week_2/Calibrations/226Ra480w2_ch000.txt',
    skiprows=4).T

# Shaping the data
ch, co = get_data(file)
ch = ch[10:5000]
co = co[10:5000]

fig, ax = plt.subplots(figsize=(10,10))
# For plotting the spectrum
ax.plot(ch, co)
ax.grid()

# The table of abundances
Abundances = {186: 3.59, 241.9: 7.43, 295.2: 19.3, 351.9: 37.6, 609.3: 46.1, 768.4: 4.94, 934.1: 3.03,
              1120.3: 15.1, 1238.1: 5.79, 1377.7: 4, 1764.5: 15.4, 2204.1: 5.08, 2447.7: 1.57}
ab_errs = [6e-2, 11e-2, 2e-1, 4e-1, 5e-1, 6e-2, 4e-2, 2e-1, 8e-2, 6e-2, 2e-1, 4e-2, 2e-2]
# Our initial guesses
p_inits = [[3100, 186, 3.5, 300],
           [3000, 241.9, 3.5, 1900 / 10],
           [7000, 295.2, 3.5, 150],
           [11500, 351.9, 3.5, 100],
           [7100, 609.3, 3.5, 50],
           [500, 768.4, 3.5, 20],
           [200, 934.1, 3.5, 10],
           [1000, 1120.3, 3.5, 5],
           [400, 1238.1, 3.5, 10],
           [300, 1377.7, 3.5, 10],
           [500, 1764.5, 20, 5],
           [200, 2204.1, 8, 0],
           [50, 2447.7, 8, 0]]
fit_withs = [10,15,15,15,15,10,15,20,20,20,25,30,30]
p_opts = np.array([])
p_covs = np.array([])

#ax.plot(ch, gauss(ch,*p_inits[-1]))
# Fitting the Gaussians to the spectrum
for w,p in zip(fit_withs,p_inits):
    # Getting an appropriate range to fit to:
    xs = ch[(ch > p[1] - w) & (ch < p[1] + w)]
    ys = co[(ch > p[1] - w) & (ch < p[1] + w)]
    # Fitting:
    y_err = np.sqrt(ys)
    y_err[np.where(y_err == 0)] = 1
    p_opt, p_cov = curve_fit(gauss, xs, ys, p, y_err, bounds=(0, np.inf))
    p_cov = np.sqrt(np.diag(p_cov))
    p_opts = np.vstack((p_opts, p_opt)) if p_opts.size else p_opt
    p_covs = np.vstack((p_covs, p_cov)) if p_covs.size else p_cov
    # For plotting the fit
    ax.plot(xs, gauss(xs, *p_opt))

fig.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))

# Getting our variables for the area of the gaussians with their uncertanties
sigmas = p_opts.T[2]
sigma_err = p_covs.T[2]
heights = p_opts.T[0]
heights_err = p_covs.T[0]
xs = np.array(list(Abundances.keys()))

# Finding N_measured and errors
N_measured = np.array(np.sqrt(2 * np.pi) * np.abs(sigmas) * heights)
N_exp = np.array([Abundances[k] / 100 * Dt * activity for k in list(Abundances.keys())])
N_m_errs = N_measured*np.sqrt((sigma_err/sigmas)**2+(heights_err/heights)**2)

# Converting equation so that it's easier to work with
ys = N_measured / N_exp

ys_err = np.sqrt(2*np.pi)*100/(activity*Dt*np.array([Abundances[k] for k in Abundances.keys()]))*np.sqrt([(p_opts.T[0]*np.sqrt(p_covs.T[2]))**2 + (p_opts.T[2]*np.sqrt(p_covs.T[0])**2)])
ys_err = ys_err[0]

# For plotting the epsilon(E) curve
ax.errorbar(list(Abundances.keys()), ys, xerr=p_opts.T[2], yerr=ys_err, fmt='k.', ms=1, label='Data',
            capsize=2, lw=1.2)

# Prettyfying the plot
ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(which='both')
fig.suptitle('Efficiency')
ax.set_xlabel('Energy [eV]')
ax.set_ylabel(r'Efficiency, $\epsilon = \frac{dN_{measured}}{dN_{expected}}$')


# The function to fit with
potens = lambda x, a, b: a * x ** b
p_init = [2e-8, -1]

# Fitting the function
p_opt, p_cov = curve_fit(potens, xs, ys, p_init, ys_err)

# Plotting the fit
ax.plot(xs, potens(xs, *p_opt), 'r-', label='Fit')

# Writing the function in the plot
ax.text(450, 0.0075, r'$\epsilon(E)\simeq ({a}\pm{da})\cdot E^{{{b}\pm{db}}}$'.format(a=p_opt[0].round(3),
                                                                                       b=p_opt[1].round(4),
                                                                                       da=np.sqrt(p_cov[0, 0]).round(3),
                                                                                       db=np.sqrt(p_cov[1, 1]).round(4)))

print(f'epsilon = ({p_opt[0]} +- {np.sqrt(p_cov[0,0])})*E^({p_opt[1]}+-{np.sqrt(p_cov[1,1])})')
ax.legend()
fig.tight_layout()
plt.show()
fig.savefig('../Efficiency.pdf')



