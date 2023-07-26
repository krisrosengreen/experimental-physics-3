import numpy as np
from numpy import exp, log, sqrt, cos, sin, pi
from numpy import append as app
from numpy import array as arr
import matplotlib.pyplot as plt
import helpers.mcaloader as mca
from helpers.data import DataStructure as DS
import os
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import scipy.constants as cs
# Til at fitte og se bort fra outliers samtidigt
from sklearn.linear_model import HuberRegressor


np.seterr(divide='ignore')  # Slip af med "Divide by zero error"


def gauss(x, H, A, x0, sigma):
    """
    Gaussian function used for fitting peaks.
    """
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def keV2wvl(E):
    """
    Given energy E in keV return the associated wavelength in nm

    Args:
        E: Energy in keV

    returns wavelength of given energy in nm
    """
    wvl = cs.Planck*cs.c/(E*1000*cs.eV)*10**9
    return wvl


class AnalyzeMain:
    def __init__(self, kwargs):
        self.data_used = {}
        self.kwargs = kwargs
        self.title_args = None

        xscale = 1
        yscale = 1

    def set_plt_titles(self, title_args):
        self.title_args = title_args

    def quick_plot(self, xs, ys, title="", clear_figure=True):
        def funlin(x, a, b):
            return a*x+b

        if self.kwargs["log"]:
            plt.yscale("log")

        if not self.kwargs["scatter"]:
            plt.plot(xs, ys)
        else:
            plt.scatter(xs, ys)
        plt.title(title)

        if self.kwargs["xlim"] != None:
            xlim = list(map(int, self.kwargs["xlim"].split(',')))

            plt.xlim(xlim)

        if self.title_args != None:
            plt.title(self.title_args["title"])
            plt.xlabel(self.title_args["xlabel"])
            plt.ylabel(self.title_args["ylabel"])

        if self.kwargs["fit"]:
            def funlin(x, a, b):
                return a*x + b

            xm = xs.reshape(-1, 1)

            model = HuberRegressor().fit(xm, ys)
            b = model.intercept_
            a = model.coef_[0]

            popt, _ = curve_fit(funlin, xs, ys)
            ls = np.linspace(min(xs), max(xs))
            plt.plot(ls, funlin(ls, a, b),
                     label=rf"Fit: $f(x)={a:.4f}\cdot x+{b:.4f}$", c='m')
            plt.legend()

        if self.kwargs["savefig"] != None:
            plt.savefig("Figures/"+self.kwargs["savefig"])
            print("Figure saved!")

        if not self.kwargs["hide"]:
            plt.show()
        elif clear_figure:
            plt.clf()
            plt.cla()

    def plot_file(self, file):
        """
        Plot file

        Args:
            file: path+name to file
            log: Whether the y-data should be logarithmic
        """
        data = mca.load_mca(file)
        xs = np.arange(0, len(data), 1)
        name = file.split('/')[-1].split('.')[0]

        self.data_used["xs"] = xs
        self.data_used["ys"] = data

        self.quick_plot(xs*self.xscale, data*self.yscale, name)

    def plot_folder(self, folder):
        """
        Given a folder name, plot all files from within that folder:

        Args:
            folder: str to folder
            log: Whether the y-data should be logarithmic
        """

        for filename in os.listdir(folder):
            self.plot_file(folder + filename)

    def get_troughs_from_index(self, index, ys):
        """
        Given an index from data, of where a peak is. Get the index of two points that
        contitute the troughs of the peak

        Args:
            index: The index value of a peak
            data: The y-data corresponding to a histogram.

        returns list: List containing the index of the lower and upper troughs
        """
        data_mean = np.mean(ys)
        peak_value = ys[index]
        threshold = peak_value - (peak_value-data_mean)*2/3

        def getindex(step):
            i = index + step
            cindex = None
            while True:
                if ys[i] < threshold:
                    cindex = i
                    break

                i += step
            return cindex

        return [getindex(-10), getindex(10)]

    def fit_to_peak(self, xs, ys, index_peak):
        """
        Given data and index to a peak to fit, troughs around the peak will be found,
        and gaussian function will be fittet in that range.

        Args:
            data: The y-data corresponding to a histogram.
            index_peak: Index to the position of the peak value in the data list

        returns params, troughs, xdata and ydata for the range
        """
        troughs_index = self.get_troughs_from_index(index_peak, ys)
        xdata = xs[troughs_index[0]:troughs_index[1] + 1]
        ydata = ys[troughs_index[0]:troughs_index[1] + 1]

        params, pcov = curve_fit(gauss, xdata, ydata, p0=[
            min(ydata), max(ydata), xs[index_peak], 1])

        return params, (troughs_index, xdata, ydata, pcov)

    def plot_peaks_with_gauss(self, xs, ys, peaks, name):
        """
        Given a peak, plot the data given around peak and a gaussed function
        fitted to the peak, along with printing the value corresponding to the peak
        of the gaussian function.

        Args:
            data: List of y-data corresponding to a histrogram
            peaks: List of indeces of peaks.
        """
        hide_plot = self.kwargs["hide"]
        scatter = self.kwargs["scatter"]

        l = len(peaks)
        sqrtl = int(np.sqrt(l))
        dister1 = sqrtl if sqrtl**2 >= l else sqrtl + \
            1  # int(math.sqrt(len(peaks)) // 1 + 1)
        dister2 = sqrtl if len(peaks)/sqrtl > sqrtl - 1 else sqrtl - 1
        dister2 = dister2 if dister1 * dister2 >= l else dister2 + 1
        fig, axes = plt.subplots(dister1, dister2)
        fig.suptitle(name)
        axes = axes.reshape(dister1 * dister2)

        xtops = []
        ytops = []
        xtops_errors = []

        for c, peak in enumerate(peaks):
            params, (troughs, xdata, ydata, pcov) = self.fit_to_peak(xs, ys, peak)
            errors = np.sqrt(np.diag(pcov))


            params2, pcov2 = curve_fit(gauss, xdata*self.xscale, ydata, p0=[
                min(ydata), max(ydata), xs[peak]*self.xscale, 0.1])
            errors2 = np.sqrt(np.diag(pcov2))


            if not scatter:
                axes[c].plot(xdata * self.xscale, ydata)
            if scatter:
                axes[c].scatter(xdata * self.xscale, ydata)

            axes[c].plot(xdata * self.xscale,
                         [gauss(i, *params2) for i in xdata*self.xscale])
            axes[c].scatter([xs[i] * self.xscale for i in troughs], [ys[i]
                            for i in troughs], c=['g'] * len(troughs))
            axes[c].grid(True)
            axes[c].scatter([params[2] * self.xscale],
                            [gauss(params[2], *params)], c='r')

            ytop = gauss(params[2], *params)
            print(f"At peak: {peak}", f"x_top is: {params[2]:.5f}",
                  f"y_top is: {ytop:.5f}", f"energy: {params[2]*self.xscale} keV", f"x_top sigma: {params2[3]:.2f}")
            
            xtops.append(params[2] * self.xscale)
            ytops.append(ytop)
            xtops_errors.append(params2[3])

        print()

        self.data_used["xtops"] = xtops
        self.data_used["ytops"] = ytops
        self.data_used["xtops_errors"] = xtops_errors

        for axis in axes[len(peaks):]:
            axis.axis('off')
        if not hide_plot:
            fig.show()

    @staticmethod
    def peaks_list(data):
        """
        Given data, return list of peaks found

        Args:
            data: List of y-data corresponding to a histrogram

        returns list of peak indeces
        """
        return find_peaks(data, prominence=5, distance=20,
                          height=max(data) / 10)[0]

    def plot_file_with_gauss(self, file):
        """
        Given a file, plot the file and plot peaks fitted with a gaussian function.

        Args:
            file: path+filename of file to load (mca format).
            hide_plot: Whether to hide the plots and just print the text instead.
        """
        hide_plot = self.kwargs["hide"]
        name = file.split("/")[-1]

        print(f" >>> {name}")
        ys = mca.load_mca(file)
        xs = np.arange(0, len(ys), 1)

        f = plt.figure(0)
        peaks = self.peaks_list(ys)
        plt.scatter([xs[i] * self.xscale for i in peaks], [ys[i]
                    for i in peaks], c=['r'] * len(peaks))

        plt.grid()
        title = name.split(".")[0]

        if not hide_plot:
            f.show()

        self.quick_plot(xs*self.xscale, ys*self.yscale, title)
        self.plot_peaks_with_gauss(xs, ys, peaks, name.split(".")[0])

        if not hide_plot:
            plt.show()

    def plot_folder_with_gauss(self, folder):
        """
        Given a folder name plot all files in them, find peaks. Then fit
        gaussian to the peaks and find the top point of that gaussian
        fitted function.

        Args:
            folder: Name of plot
            hide_plot: Whether or not to hide the plots found and just print results instead
        """

        for name in os.listdir(folder):
            self.plot_file_with_gauss(folder + name)

    def tempfile(self, args):
        """
        Handle temporary files. These files are created when the '-save' flag is added. The way this is done is by calling
        >>> py main.py --tempfile <options> x=<x-data expression> y=<y-data expression>

        <options>:
        1. Either name of file without the ending. I.e. if file is called "temp.txt", just write "temp" and all
           data columns will be loaded as "<name of file>_<name of column>". I.e. "temp_xs"
        2. Explicit column is importet from a file and named a certain way.
           Format is <local variable name>:<name of file>:<name of column in file>
           Example: xdata:temp:xs

        <x-data expression>: Variables are enclosed within dollar symbols.
        1. log($temp_xs$/$temp2_xs$)

        <y-data expression>: similar to <x-data expression>

        """
        vardict = {}

        current_index = 0

        # Load variables
        x = None
        y = None

        while current_index < len(args):
            strinit = args[current_index][:2]
            if strinit == "x=" or strinit == "y=":
                execx_str = args[current_index].split('=')[1]

                for vname in vardict.keys():
                    execx_str = execx_str.replace(
                        f"${vname}$", f"vardict['{vname.replace('$', '')}']")

                if strinit == "x=":
                    x = eval(execx_str)
                else:
                    y = eval(execx_str)
            else:
                if ':' in args[current_index]:
                    variable_args = args[current_index].split(':')
                    variable_name = variable_args[0]
                    variable_fromfile = variable_args[1]
                    variable_columnname = variable_args[2]

                    dat = DS()
                    dat.loadfile(variable_fromfile+".txt")
                    vardict[variable_name] = dat.df[variable_columnname].to_numpy()
                else:
                    tempname = args[current_index]
                    dat = DS()
                    dat.loadfile(tempname+".txt")
                    for key in dat.df.keys():
                        vardict[tempname+"_"+key] = dat.df[key].to_numpy()

            current_index += 1

        self.quick_plot(x, y)

    def save_data_used(self, filename):
        """
        Save data to a specified file.

        Args:
            filename: Name of file to save
        """
        DS.savedict(self.data_used, filename)


class AnalyzeEX2(AnalyzeMain):
    def __init__(self, kwargs):
        super().__init__(kwargs)

        # CALIBRATION START
        def funlin(x, a, b):
            return a*x + b

        Am241_xs = np.array([900.257, 1091.514, 1145, 1340.27])
        Am241_tabel_energy = np.array([13.9, 16.84, 17.7502, 20.7848])

        Fe55_xs = np.array([382.27, 420.12])
        Fe55_tabel_energy = np.array([(6.404+6.391)/2, 7.058])

        popt, _ = curve_fit(funlin, np.concatenate(
            (Fe55_xs, Am241_xs)), np.concatenate((Fe55_tabel_energy, Am241_tabel_energy)))

        self.xscale = popt[0]
        self.yscale = 1
        # CALIBRATION END

    def compare_peaks(self, file1, file2, diff=5):
        """
        Compare peaks between two files, file1 and file2. Prints peak pairs that are close,
        and their difference. Lastly, this prints the peaks that could not be associated with
        any peak from the other file.

        Args:
            file1: File to be compared to file2
            file2: File to be compared to file1
        """
        file1_data = mca.load_mca(file1)
        file2_data = mca.load_mca(file2)

        index_peaks1 = self.peaks_list(file1_data)
        index_peaks2 = self.peaks_list(file2_data)

        def getxtop(data, index_peaks):
            L = []
            for index_peak in index_peaks:
                params, _ = self.fit_to_peak(data, index_peak)
                x_top = params[2]
                L.append(x_top)
            return L

        peaks1 = getxtop(file1_data, index_peaks1)
        peaks2 = getxtop(file2_data, index_peaks2)

        similars = []
        L_all = peaks1 + peaks2
        L_set = set(L_all)
        similars_all = []

        for peak1 in peaks1:
            for peak2 in peaks2:
                if abs(peak1-peak2) < diff:
                    similars.append((peak1, peak2))
                    similars_all.append(peak1)
                    similars_all.append(peak2)
                    break
        similars_set = set(similars_all)
        not_used = L_set - similars_set

        print()
        print(f"Comparing {file1} and {file2}")
        print(f"Found {len(similars)}/{len(L_all)//2} peak pairs")
        print("="*20+"\n")
        for c, (peak1, peak2) in enumerate(similars):
            self.data_used[f"peakpair{c}"] = [
                peak1*self.xscale, peak2*self.xscale]
            print(f"Pair {c+1}/{len(similars)}:")
            print(f"\t File 1: ")
            print(f"\t\t --> \t{peak1:.5f} channel")
            print(f"\t\t --> \t{peak1*self.xscale:.5f} keV")
            print(f"\t\t --> \t{keV2wvl(peak1*self.xscale):.5f} nm")

            print(f"\t File 2:")
            print(f"\t\t --> \t{peak2:.5f} channel")
            print(f"\t\t --> \t{peak2*self.xscale:.5f} keV")
            print(f"\t\t --> \t{keV2wvl(peak2*self.xscale):.5f} nm")

            delta = abs(peak1-peak2)
            print(f"\t Diff.:")
            print(f"\t\t --> \t{delta:.5f} channel")
            print(f"\t\t --> \t{delta*self.xscale:.5f} keV")
            print(
                f"\t\t --> \t{abs(keV2wvl(peak1*self.xscale)-keV2wvl(peak2*self.xscale)):.5f} nm")
            print()
        if len(not_used) != 0:
            self.data_used["peaknu"] = list(not_used)
            print("Dissimilar peaks:")
            for peak in not_used:
                print(f"\t Not found: \t{peak:.5f}")


# Deprecation
Analyze = AnalyzeEX2
