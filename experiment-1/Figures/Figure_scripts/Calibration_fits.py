"""

Author: Andreas Niebuhr

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from matplotlib.offsetbox import AnchoredText
import os

path = str(os.path.abspath(__file__)).replace('\\Figures\\Figure_scripts\\Calibration_fits.py', '')
datpath = path + '\\Data'
figpath = path + '\\Figures\\'


class DoCalibration:
    def __init__(self, name: str, plot_guess: bool = False, savefig: bool = False, ev_scale:float = 1):
        """

        :param name: Choose from list:
                        The name of the sample you want to fit
        :param savefig: bool: Default = False:
            If you want to save the figure. The name is hardcoded.

        """
        self.name = name
        self.savefig = savefig
        self.plot_guess = plot_guess
        self.scale = ev_scale
        self._funcs = {'Co w1': self._co_w1, 'Cs w1': self._cs_w1, 'Ra w1': self._ra_w1,
                       'Co w2': self._co_w2, 'Cs w2': self._cs_w2, 'Ra w2': self._ra_w2}
        try:
            self.na, self.da, self.pi, self.li, self.fi = self._funcs[self.name]()
        except KeyError:
            print(f"The name '{self.name}' is not defined. Choose from {self._funcs.keys()}")
        self.ch, self.co = self.get_data(self.da)
        self.plot_data(self.na, self.ch, self.co, self.pi, self.li, self.fi, self.savefig)
        plt.show()

    def _co_w1(self):
        """

        Get the variables for Co 60 w1
        :return:
            name: str: The title to display in the plot
            data: numpy.array[time,channel]: with the appropriate data
            p_inits: list[list] The initial guesses for the params of the Gauss functions to fit to
            lims: dict[tuple]: the x- and y-limits for the plot
            filename: str: the name to save the file to

        """
        name = r'Ch-eV Calibration $^{60}$Co'
        filename = 'Co_60_ch_ev_calib_w1'
        data = np.loadtxt(
            datpath + '\\Week_1\\Calibrations\\cal_60Co_ch000.txt',
            skiprows=5).T
        p_inits = [[145, 1585 * self.scale, 10, 0],
                   [110, 1800 * self.scale, 10, 0]]
        lims = {'xlim': (1500, 2000), 'ylim': (-5, 170)}
        return name, data, p_inits, lims, filename

    def _cs_w1(self):
        """

        Get the variables for Cs 137 w1
        :return:
            name: str: The title to display in the plot
            data: numpy.array[time,channel]: with the appropriate data
            p_inits: list[list] The initial guesses for the params of the Gauss functions to fit to
            lims: dict[tuple]: the x- and y-limits for the plot
            filename: str: the name to save the file to

        """
        name = r'Ch-eV Calibration $^{137}$Cs'
        filename = 'Cs_137_ch_ev_calib_w1'
        data = np.loadtxt(
            datpath + '\\Week_1\\Calibrations\\cal_137Cs_ch000TEST.txt',
            skiprows=5).T
        p_inits = [[4500, 900 * self.scale, 10, 0]]
        lims = {'xlim': (600, 1000), 'ylim': (-5, 6000)}
        return name, data, p_inits, lims, filename

    def _ra_w1(self):
        """

        Get the variables for Cs 137 w1
        :return:
            name: str: The title to display in the plot
            data: numpy.array[time,channel]: with the appropriate data
            p_inits: list[list] The initial guesses for the params of the Gauss functions to fit to
            lims: dict[tuple]: the x- and y-limits for the plot
            filename: str: the name to save the file to

        """
        name = r'Ch-eV Calibration $^{226}$Ra'
        filename = 'Ra_226_ch_ev_calib_w1'
        data = np.loadtxt(
            datpath + '\\Week_1\\Calibrations\\cal_226Ra_ch000.txt',
            skiprows=5).T
        p_inits = [[2000, 251, 5, 2100],
                   [3000, 326, 5, 1900],
                   [7000, 399, 5, 1500],
                   [11500, 475, 5, 1000],
                   [7100, 823, 5, 500],
                   [500, 1036, 5, 200],
                   [200, 1260, 5, 100],
                   [1000, 1515, 5, 50],
                   [400, 1673, 5, 100],
                   [300, 1862, 5, 100],
                   [150, 2340, 5, 50],
                   [500, 2388, 5, 50],
                   [100, 2498, 5, 25],
                   [50, 2865, 5, 0],
                   [150, 2980, 5, 0],
                   [50, 3310, 5, 0]]
        lims = {'xlim': (0, 3500), 'ylim': (-5, 15000)}
        for i in range(len(p_inits)):
            p_inits[i][1] *= self.scale
        return name, data, p_inits, lims, filename

    def _co_w2(self):
        """

        Get the variables for Co 60 w1
        :return:
            name: str: The title to display in the plot
            data: numpy.array[time,channel]: with the appropriate data
            p_inits: list[list] The initial guesses for the params of the Gauss functions to fit to
            lims: dict[tuple]: the x- and y-limits for the plot
            filename: str: the name to save the file to

        """
        name = r'Ch-eV Calibration $^{60}$Co'
        filename = 'Co_60_ch_ev_calib_w2'
        data = np.loadtxt(
            datpath + '\\Week_2\\Calibrations\\60Cow2_ch000.txt',
            skiprows=5).T
        p_inits = [[145, 1585 * self.scale, 10, 0],
                   [110, 1800 * self.scale, 10, 0]]
        lims = {'xlim': (1500, 2000), 'ylim': (-5, 170)}
        return name, data, p_inits, lims, filename

    def _cs_w2(self):
        """

        Get the variables for Cs 137 w1
        :return:
            name: str: The title to display in the plot
            data: numpy.array[time,channel]: with the appropriate data
            p_inits: list[list] The initial guesses for the params of the Gauss functions to fit to
            lims: dict[tuple]: the x- and y-limits for the plot
            filename: str: the name to save the file to

        """
        name = r'Ch-eV Calibration $^{137}$Cs'
        filename = 'Cs_137_ch_ev_calib_w2'
        data = np.loadtxt(
            datpath + '\\Week_2\\Calibrations\\137Csw2_ch000.txt',
            skiprows=5).T
        p_inits = [[4500, 900 * self.scale, 10, 0]]
        lims = {'xlim': (600, 1000), 'ylim': (-5, 10000)}
        return name, data, p_inits, lims, filename

    def _ra_w2(self):
        """

        Get the variables for Cs 137 w1
        :return:
            name: str: The title to display in the plot
            data: numpy.array[time,channel]: with the appropriate data
            p_inits: list[list] The initial guesses for the params of the Gauss functions to fit to
            lims: dict[tuple]: the x- and y-limits for the plot
            filename: str: the name to save the file to

        """
        name = r'Ch-eV Calibration $^{226}$Ra'
        filename = 'Ra_226_ch_ev_calib_w2'
        data = np.loadtxt(
            datpath + '\\Week_2\\Calibrations\\226Raw2_ch000.txt',
            skiprows=5).T
        p_inits = [[2000, 251, 5, 2100],
                   [3000, 326, 5, 1900],
                   [7000, 399, 5, 1500],
                   [11500, 475, 5, 1000],
                   [7100, 823, 5, 500],
                   [500, 1036, 5, 200],
                   [200, 1260, 5, 100],
                   [1000, 1515, 5, 50],
                   [400, 1673, 5, 100],
                   [300, 1862, 5, 100],
                   [150, 2340, 5, 50],
                   [500, 2388, 5, 50],
                   [100, 2498, 5, 25],
                   [50, 2865, 5, 0],
                   [150, 2980, 5, 0],
                   [50, 3310, 5, 0]]
        lims = {'xlim': (0, 3500), 'ylim': (-5, 28000)}
        for i in range(len(p_inits)):
            p_inits[i][1] *= self.scale
        return name, data, p_inits, lims, filename

    def get_data(self, data: list[list]):
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

        return channels * self.scale, counts

    @staticmethod
    def gauss(x, *p):
        """

        :param x: array of x-values
        :param p: array of [a,b,c] where
            a = the height of the gaussian
            b = the center_x of the gaussian
            c = 2 * the width of the gaussian
        :return: a gaussian of a * exp(-[(b - x) / c]**2) + d
        """
        a, b, c, d = p
        return a * np.exp(-((b - x) / c) ** 2) + d

    def plot_data(self, name: str, channels: np.ndarray, counts: np.ndarray, p_inits: list[list], lims: dict[str:tuple],
                  filename: str, savefig: bool):
        """

        Fit and plot the data

        :param name: str: The title of the plot
        :param channels: numpy.ndarray: The channel data
        :param counts: numpy.ndarray: The counts data
        :param p_inits: list[list]: A list with the different p_init options for the fit
        :param lims: dict[str:tuple]: Where 'xlim': the limits on the x-axis and 'ylim': the limits on the y-axis
        :param filename: str: The save-name of the file
        :param savefig: bool: If you want to save the file
        :return: Shows and saves a plot to /Figures/<filename>.pdf with data, fit and fitted center-x and height
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.suptitle(name)
        # Plotting the data with an error of +-0.5channels and +-sqrt(N) counts
        ax.errorbar(channels, counts, yerr=np.sqrt(counts), xerr=.5, fmt='k', ls='None', marker='None', capsize=2,
                    label='Data')
        # Setting an appropriate limit for the plot
        ax.set_xlim(lims['xlim'])
        ax.set_ylim(lims['ylim'])

        # Loop over the guesses
        text = []
        for n, p in enumerate(p_inits):
            # Setting an appropriate range of channels and counts
            xs = channels[(channels > p[1] - 25) & (channels < p[1] + 25)]
            ys = counts[(channels > p[1] - 25) & (channels < p[1] + 25)]
            if self.plot_guess:
                ax.plot(xs, self.gauss(xs, *p), 'r:')
            # Fitting the curve
            p_opt, p_cov = curve_fit(self.gauss, xs, ys, p, np.sqrt(np.sqrt(ys) ** 2 + 0.25), True)
            # Plotting the curve
            ax.plot(xs, self.gauss(xs, *p_opt), zorder=10, label=f'Fit - Peak {n + 1}')
            # Text to show the found center-channel and height of the peaks
            dh = '\infty' if np.isinf(p_cov[0, 0]) else p_cov[0, 0].round(3)
            text.append(r'Peak{n}: $x = {x} \pm {dx}, h = {h} \pm {dh}$'.format(n=n + 1, x=p_opt[1].round(3),
                                                                                dx=p_opt[2].round(3),
                                                                                h=p_opt[0].round(3),
                                                                                dh=dh))
            print(f'x_err = {p_opt[2]}')
        # Making all the texts into a single string
        plottext = ''
        for i in range(len(text)):
            if i == len(text) - 1:
                plottext += text[i]
            else:
                plottext += text[i] + '\n'
        # Plotting the text as a textbox
        textbox = AnchoredText(plottext, 'upper center')
        ax.add_artist(textbox)

        # Prettifying the plot
        ax.legend()
        ax.set_xlabel('Channel')
        ax.set_ylabel('Counts')
        ax.grid()
        fig.tight_layout()
        fig.show()
        # Saving and showing the plot
        if savefig:
            fig.savefig(figpath + filename + '.pdf')


DoCalibration('Co w2')



######### Week 1 #########


Co_lines = [1173.228, 1332.492]
Cs_lines = [661.657]
Ra_lines = [186.211, 241.997, 295.224, 351.932, 609.312, 768.365, 934.061, 1120.287, 1238.110, 1377.669, 1729.595,
            1764.494, 1847.420, 2118.55, 2204.21, 2447.86]

Co_channels = [1584.347, 1799.565]
Cs_channels = [892.716]
Ra_channels = [251.472, 326.909, 398.994, 475.712, 823.973, 1039.273, 1263.142, 1515.058, 1674.299, 1864.261, 2339.364,
               2385.89, 2496.495, 2863.276, 2979.556, 3308.734]

lines = np.append(np.append(Co_lines, Cs_lines), Ra_lines)
chans = np.append(np.append(Co_channels, Cs_channels), Ra_channels)

sl = [3e-4, 3e-4, 3e-4, 1.3e-4, 3e-4, 2e-4, 2e-4, 7e-4, 1e-4, 1.2e-4, 1e-4, 1.2e-4, 1.2e-4, 1.5e-4, 1.4e-4, 2.5e-4,
      3e-3, 4e-3, 1e-3]
sc = [.021, .024, 0, .003, .001, 0, 0, 0, .008, .026, .004, .017, .039, .079, .008, .125, .233, .039, .132]

sigmas = lines / chans * np.sqrt((sl / lines) ** 2 + (sc / chans)**2)
sigma = np.sqrt(np.sum(sigmas ** 2))

ev_pr_ch = np.mean([l / c for l, c in zip(lines, chans)])
print(f'eV/Ch_w1 = {ev_pr_ch} +- {sigma}')


######### Week 2 #########


Co_channels = [1584.82, 1800.105]
Cs_channels = [893.039]
Ra_channels = [250.221, 325.82, 397.927, 474.64, 822.818, 1038.034, 1261.767, 1513.848, 1673.22, 1863.203, 2338.027,
               2384.712, 2495.609, 2862.051, 2978.132, 3307.96]

chans1 = np.append(np.append(Co_channels,Cs_channels),Ra_channels)

sc1 = [8.888959092440789, 9.086069796355481, 6.468434103852279, 5.647174062324962, 5.312653544161381, 5.64767039522953, 5.796441012090918, 6.536327833283606, 6.720850920786223, 7.58068349702447, 8.611302763966878, 9.003024832352258, 10.572325137458167, 10.393217795937248, 12.381648184882081, 14.28120623208946, 16.24559650115866, 17.802392544631125, 18.755349858456835]

sigmas1 = lines / chans1 * np.sqrt((sl / lines) ** 2 + (sc1 / chans1)**2)
sigma1 = np.sqrt(np.sum(sigmas1 ** 2))
ev_pr_ch = np.mean([l / c for l, c in zip(lines, chans1)])
print(f'eV/Ch_w2 = {ev_pr_ch} +- {sigma1}')


######### Combining the two #########


chans = np.append(chans, chans1)
lines = np.append(lines, lines)
sigma = np.sqrt(sigma**2 + sigma1**2)
ev_pr_ch = np.mean([l / c for l, c in zip(lines, chans)])
print(f'eV/Ch_combined = {ev_pr_ch} +- {sigma}')


