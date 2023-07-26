import sys
sys.path.append("../")

from helpers.mcaloader import load_mca as mca
import helpers.pathfinder as pf
import matplotlib.pyplot as plt
import numpy as np
import analyzepeaks as ap
from scipy.optimize import curve_fit


def gauss(x, H, A, x0, sigma):
    """
    Gaussian function used for fitting peaks.
    """
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

class Meteorites:
    def __init__(self):
        self.files = pf.dir_file_crawler("data/")
        self.getfile = lambda filename: pf.getfile_in_L(filename, self.files)
        self.full_filenames = lambda filenames: list(map(self.getfile, filenames))
        self.getdata = lambda filenames: list(map(mca, self.full_filenames(filenames)))

    def meteorite_reference(self):
        file_peaks = self.measured_meteorites()

        # Reference
        filenames = ["36KV50Few3", "36KV64Few3"]
        datas = self.getdata(filenames)

        max_val = 0
        for data in datas:
            if max(data) > max_val:
                max_val = max(data)

        third_peaks = []
        first_peaks = []
        file_scales = []

        for data, filename in zip(datas, filenames):
            try:
                yscale = max_val/max(data)
                xs = np.arange(0, len(data))
                ys = data*yscale

                analyzer = ap.Analyze({})

                peaks_index = analyzer.peaks_list(ys)
                peaks_xs = xs[peaks_index]
                peaks_ys = ys[peaks_index]/max_val
                ys/=max_val

                print(peaks_xs, peaks_ys)

                plt.scatter(peaks_xs*ap.xscale, peaks_ys)

                plt.plot(xs*ap.xscale, ys, label=filename)

                file_scales.append(yscale)
                first_peaks.append(peaks_ys[0]/yscale)

                third_peaks.append(peaks_ys[2])
            except:
                print("Exception occurred at", filename)

        for i in third_peaks[2:]:
            print("Ratio: ", third_peaks[0]/i)
            self.ratio = third_peaks[0]/i

        plt.xlim(6,8)
        plt.legend()
        plt.grid()
        plt.show()

    def measured_meteorites(self):
        filenames = ["36KVDiablow3", "36KVShiningw3", "36KVSpottedw3"]
        datas = self.getdata(filenames)

        file_peaks = {}

        for data, filename in zip(datas, filenames):
            xs = np.arange(0, len(data))
            index_lower = np.where(xs > 470)[0][0]
            index_upper = np.where(xs > 500)[0][0]
            plt.scatter(xs[index_lower:index_upper], data[index_lower:index_upper], s=5, label=filename)
            popt, pcov = curve_fit(gauss, xs[index_lower:index_upper], data[index_lower:index_upper], p0=[0,25,485,1])
            # def gauss(x, H, A, x0, sigma):

            ls = np.linspace(xs[index_lower], xs[index_upper])
            ys = gauss(ls, *popt)
            plt.plot(ls, ys, label=filename + " w/ gauss")

            print(filename, popt[1], np.sqrt(np.diag(pcov))[1])

            file_peaks[filename] = popt[1]

        # plt.legend()
        # plt.grid()
        # plt.show()

        return file_peaks


if __name__ == "__main__":
    M = Meteorites()
    M.measured_meteorites()
    #M.meteorite_reference()