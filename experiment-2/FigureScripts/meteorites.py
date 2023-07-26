import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so
import scipy.signal as ss
from matplotlib.patches import ConnectionPatch

xscale = 0.014866456685197033


def load_mca(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

        start = None
        end = None

        for i in range(len(lines)):
            if lines[i] == "<<DATA>>\n":
                start = i
            elif lines[i] == "<<END>>\n":
                end = i
                break

        data_lines = lines[start + 1:end]
        processed_lines = []

        for i in data_lines:
            processed_lines.append(int(i.strip()))

        return processed_lines

cal5050 = np.array(load_mca('C:/Users/andre/PycharmProjects/EF3/eksperiment-2/data/Dataw3/36KV50Few3.mca'))
cal6446 = np.array(load_mca('C:/Users/andre/PycharmProjects/EF3/eksperiment-2/data/Dataw3/36KV64Few3.mca'))

max50 = max(cal5050)
max64 = max(cal6446)
# cal5050 = cal5050/max50
# cal6446 = cal6446/max64
xs = np.arange(0, len(cal5050), 1) * xscale

# cal5050 = cal5050/max(cal5050)
# cal6446 = cal6446/max(cal6446)


# %%
fig, ax = plt.subplots()

ax.set_xlim(5.5, 7.5)


def func(x, *p):
    a, b, c, d, e, f, g, h, i = p
    return a * np.exp(-((b - x) / c) ** 2) + d * np.exp(-((e - x) / f) ** 2) + g * np.exp(-((h - x) / i) ** 2)


def func1(x, *p):
    a, b, c = p
    return a * np.exp(-((b - x) / c) ** 2)


p_init = [750, 6.16, .08, 100, 6.8, .09, 150, 7.2, .08]
p_init1 = [2100, 6.15, .08, 400, 6.75, .08, 300, 7.25, .06]
# ax.plot(xs, func(xs, *p_init1),'r-')
# plt.plot(xs,func(xs,*p_init1))


p_opt50, p_cov50 = so.curve_fit(func, xs, cal5050 + 1, p_init, sigma=np.sqrt(cal5050 + 1), absolute_sigma=True)
p_opt64, p_cov64 = so.curve_fit(func, xs, cal6446 + 1, p_init1, sigma=np.sqrt(cal6446 + 1), absolute_sigma=True)
ax.errorbar(xs, cal5050 / max(func(xs, *p_opt50)), yerr=np.sqrt(cal5050) / max(func(xs, *p_opt50)), fmt='.',
            color='darkcyan', label='50/50', capsize=2)
ax.errorbar(xs, cal6446 / max(func(xs, *p_opt64)), yerr=np.sqrt(cal6446) / max(func(xs, *p_opt64)), fmt='.',
            color='darkgreen', label='64/36', capsize=2)
print(p_opt64)
# print(p_opt64)
# ax.plot(xs,func1(xs,*p_init1),label='Fit')
# ax.plot(xs,func(xs,*p_init))
ax.plot(xs, func(xs, *p_opt50) / max(func(xs, *p_opt50)), 'blue', lw=.7, label='fit 50/50')
ax.plot(xs, func(xs, *p_opt64) / max(func(xs, *p_opt64)), 'yellowgreen', lw=.7, label='fit 64/36')
ax.legend()
ax.grid()
ax.set_ylabel('Counts')
ax.set_xlabel('Voltage [keV]')
ax.set_title('Iron/Nickel Calibrations, Normalized')

fitxs = np.linspace(0, 100, 10000)
x50, y50 = ss.find_peaks(func(fitxs, *p_opt50), 50)
y50 = y50['peak_heights']
x64, y64 = ss.find_peaks(func(fitxs, *p_opt64), 50)
y64 = y64['peak_heights']

n_peak50 = y50 / max(y50)
n_peak64 = y64 / max(y64)
calib_peaks = np.array([1 / n_peak50[2], 1 / n_peak64[2]])
calib_errors = np.array([np.sqrt((p_cov50[1, 1] / n_peak50[0]) ** 2 + p_cov50[7, 7] / n_peak50[2]) / n_peak50[2],
                         np.sqrt((p_cov64[1, 1] / n_peak64[0]) ** 2 + p_cov64[7, 7] / n_peak64[2]) / n_peak64[2]])

fig.tight_layout()
fig.show()
# fig.savefig('C:/Users/andre/PycharmProjects/EF3/eksperiment-2/Figures/Meteorites/Meteorite_Normalized_Calibration_fit.pdf')

plt.close(fig)
# %%

shining = np.array(load_mca('C:/Users/andre/PycharmProjects/EF3/eksperiment-2/data/Dataw3/36KVShiningw3.mca'))
spotted = np.array(load_mca('C:/Users/andre/PycharmProjects/EF3/eksperiment-2/data/Dataw3/36KVSpottedw3.mca'))
deathw = np.array(load_mca('C:/Users/andre/PycharmProjects/EF3/eksperiment-2/data/Dataw3/36KVDiablow3.mca'))
xs = np.arange(0, len(shining), 1) * xscale

fig, axs = plt.subplots(1, 2, figsize=(8, 5))
fig.suptitle('Meteorite spectra')
for a in axs:
    a.set_xlabel('keV')
ax = axs[0]
ax1 = axs[1]
ax.set_xlim(5.5, 7.5)
ax.set_ylabel('Counts / fitted max count')

p_init_deathv = [2200, 6.15, .08, 400, 6.8, .08, 50, 7.2, .08]
p_init_shining = [1450, 6.15, .08, 250, 6.8, .08, 50, 7.2, .08]
p_init_spotted = [1100, 6.15, .08, 200, 6.8, .08, 50, 7.2, .08]

p_opt_sh, p_cov_sh = so.curve_fit(func, xs, shining, p_init_shining, np.sqrt(shining + 1), True)
p_opt_sp, p_cov_sp = so.curve_fit(func, xs, spotted, p_init_spotted, np.sqrt(spotted + 1), True)
p_opt_dv, p_cov_dv = so.curve_fit(func, xs, deathw, p_init_deathv, np.sqrt(deathw + 1), True)
ax.errorbar(xs, shining / max(func(xs, *p_opt_sh)), yerr=np.sqrt(shining) / max(func(xs, *p_opt_sh)), capsize=2,
            fmt='.', label='Shining')
ax.errorbar(xs, spotted / max(func(xs, *p_opt_sp)), yerr=np.sqrt(spotted) / max(func(xs, *p_opt_sp)), capsize=2,
            fmt='.', label='Spotted')
ax.errorbar(xs, deathw / max(func(xs, *p_opt_dv)), yerr=np.sqrt(deathw) / max(func(xs, *p_opt_dv)), capsize=2, fmt='.',
            label='DeathValley')

sh_err = np.sum([(p_cov_sh[i, i] / p_opt_sh[i]) ** 2 for i in (0, 3, 6)])
sp_err = np.sum([(p_cov_sp[i, i] / p_opt_sp[i]) ** 2 for i in (0, 3, 6)])
dv_err = np.sum([(p_cov_dv[i, i] / p_opt_dv[i]) ** 2 for i in (0, 3, 6)])
m_errs = np.array([sh_err, sp_err, dv_err])
fitxs = np.linspace(0, 10, 1000)

shining_fit = func(fitxs, *p_opt_sh)
spotted_fit = func(fitxs, *p_opt_sp)
deathv_fit = func(fitxs, *p_opt_dv)
ax.plot(fitxs, shining_fit / max(shining_fit), label='Fit Shining', color='tab:cyan')
ax.plot(fitxs, spotted_fit / max(spotted_fit), label='Fit Spotted', color='tomato')
ax.plot(fitxs, deathv_fit / max(deathv_fit), label='Fit DeathValley', color='yellowgreen')

sh_peaks_x, sh_peaks_height = ss.find_peaks(shining_fit, 20)
sh_peaks_height = sh_peaks_height['peak_heights']
sp_peaks_x, sp_peaks_height = ss.find_peaks(spotted_fit, 10)
sp_peaks_height = sp_peaks_height['peak_heights']
dv_peaks_x, dv_peaks_height = ss.find_peaks(deathv_fit, 10)
dv_peaks_height = dv_peaks_height['peak_heights']

n_sh_peaks_h = sh_peaks_height / max(sh_peaks_height)
n_sp_peaks_h = sp_peaks_height / max(sp_peaks_height)
n_dv_peaks_h = dv_peaks_height / max(dv_peaks_height)

peak_ratios = np.array([1 / n_sh_peaks_h[2], 1 / n_sp_peaks_h[2], 1 / n_dv_peaks_h[2]])

ax.legend()
ax.grid()

ax1.errorbar(xs, shining / max(func(xs, *p_opt_sh)), yerr=np.sqrt(shining) / max(func(xs, *p_opt_sh)), capsize=2,
             fmt='.', label='Shining')
ax1.errorbar(xs, spotted / max(func(xs, *p_opt_sp)), yerr=np.sqrt(spotted) / max(func(xs, *p_opt_sp)), capsize=2,
             fmt='.', label='Spotted')
ax1.errorbar(xs, deathw / max(func(xs, *p_opt_dv)), yerr=np.sqrt(deathw) / max(func(xs, *p_opt_dv)), capsize=2, fmt='.',
             label='DeathValley')
ax1.plot(fitxs, shining_fit / max(shining_fit), label='Fit Shining', color='tab:cyan')
ax1.plot(fitxs, spotted_fit / max(spotted_fit), label='Fit Spotted', color='tomato')
ax1.plot(fitxs, deathv_fit / max(deathv_fit), label='Fit DeathValley', color='yellowgreen')
ax1.set_xlim(7, 7.4)
ax1min = -0.001
ax1max = 0.03
ax1.set_ylim(ax1min, ax1max)
ax1.yaxis.tick_right()
ax1.grid(True)
ax.fill((7, 7.4, 7.4, 7), (ax1min, ax1min, ax1max, ax1max), alpha=0.5, color='tab:blue')
con1 = ConnectionPatch((7, ax1min), (7, ax1min), ax.transData, ax1.transData, alpha=0.5, color='tab:blue')
con2 = ConnectionPatch((7.4, ax1min), (7.4, ax1min), ax.transData, ax1.transData, alpha=0.5, color='tab:blue')
con3 = ConnectionPatch((7, ax1max), (7, ax1max), ax.transData, ax1.transData, alpha=0.5, color='tab:blue')
con4 = ConnectionPatch((7.4, ax1max), (7.4, ax1max), ax.transData, ax1.transData, alpha=0.5, color='tab:blue')

fig.add_artist(con1)
fig.add_artist(con2)
fig.add_artist(con3)
fig.add_artist(con4)
fig.tight_layout()
# fig.savefig('C:\\Users\\andre\\PycharmProjects\\EF3\\eksperiment-2\\Figures\\Meteorites\\Meterorites_Normalized.pdf')
fig.show()
plt.close(fig)
# %%
yerr = np.array([0.05, 0.05])
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot()
ax.errorbar(calib_peaks, [1, 64 / 36], xerr=calib_errors, yerr=yerr, fmt='k', marker='None', linestyle='None',
            capsize=2, label='Calibrations')


def linfun(x, *p):
    a, b = p
    return a * x + b


p_init = [.205, -.02]

final_p_opt, final_p_cov = so.curve_fit(linfun, calib_peaks, [1, 64 / 36], p_init,
                                        sigma=np.sqrt(calib_errors ** 2 + yerr ** 2 + 1e-5), absolute_sigma=True)

min_errors = [final_p_opt[i] - final_p_cov[i, i] for i in range(2)]
max_errors = [final_p_opt[i] + final_p_cov[i, i] for i in range(2)]
ratios = linfun(peak_ratios, *min_errors)
ratio_upper = linfun(peak_ratios + m_errs, *max_errors)
ratio_lower = linfun(peak_ratios - m_errs, *min_errors)
ratio_errs = ratio_upper - linfun(peak_ratios, *final_p_opt)
newxs = np.linspace(0, 100, 100)
ax.plot(newxs, linfun(newxs, *final_p_opt), color='k', lw=1, label='Fit')
ax.fill_between(newxs, linfun(newxs, *min_errors), linfun(newxs, *max_errors), color='r', alpha=.2, label='Fit error')

sign = '+' if final_p_opt[1] > 0 else '-'
l = np.array((5, .2))
trans_angle = ax.transData.transform_angles(np.array((41.5,)), l.reshape((1, 2)))[0]
text = plt.text(7, .2,
                r'$Fe:Ni = ({a}\pm{b})\cdot\frac{{I_{{Fe}}}}{{I_{{Ni}}}} {p} {c}\pm{d}$'.format(
                    a=final_p_opt[0].round(3),
                    b=final_p_cov[0, 0].round(3),
                    p=sign,
                    c=np.abs(final_p_opt[1].round(3)),
                    d=final_p_cov[1, 1].round(3)),
                rotation=trans_angle, rotation_mode='anchor', fontsize=11)
ax.add_artist(text)

ax.errorbar(peak_ratios, linfun(peak_ratios, *final_p_opt), xerr=m_errs, yerr=ratio_errs, fmt='r', linestyle='None',
            marker='None', label='Meteorites', capsize=2)
print(m_errs)
for p, e in zip(peak_ratios, ratio_errs):
    ax.vlines(p, 0, linfun(p, *max_errors), ls=':', color='k', lw=1, label='read-off')
    xs = np.linspace(0, p, 10)
    ax.fill_between(xs, linfun(p - e, *min_errors), linfun(p + e, *max_errors), color='k', lw=1, alpha=0.5,
                    label='read-off ')
ax.set_xlim(0, 100)
ax.set_ylim(0, 25)

ax.text(1, ratios[0] + .3, r'Shining Fe:Ni ratio = $({a}\pm{b}):1$'.format(a=ratios[0].round(3),
                                                                           b=np.round(ratio_errs[0], 3)), fontsize=10)
ax.text(1, ratios[1] + .3, r'Sparkling Fe:Ni ratio = $({a}\pm{b}):1$'.format(a=ratios[1].round(3),
                                                                             b=ratio_errs[1].round(3)), fontsize=10)
ax.text(1, ratios[2] + .3, r'DeathValley Fe:Ni ratio = $({a}\pm{b}):1$'.format(a=ratios[2].round(3),
                                                                               b=ratio_errs[2].round(3)), fontsize=10)

ax.grid()
handles, labels = ax.get_legend_handles_labels()
ha, la = [], []
for h, l in zip(handles, labels):
    if l not in la:
        la.append(l)
        ha.append(h)
yts = np.array([5, 10, 15, 20, 25])
ax.set_yticks(yts, [str(int(y)) + ':1' for y in yts])
ax.set_xlabel(r'$\frac{{I_{{Fe}}}}{{I_{{Ni}}}}$', fontsize=16)
ax.set_ylabel('Fe:Ni Ratio', fontsize=16)
ax.legend(ha, la, loc='center right')
fig.suptitle('Fe to Ni Ratio for Meteorites', fontsize=20)
fig.tight_layout()
fig.show()
fig.savefig('C:/Users/andre/PycharmProjects/EF3/eksperiment-2/Figures/Meteorites/Ratios.pdf')
plt.close(fig)
