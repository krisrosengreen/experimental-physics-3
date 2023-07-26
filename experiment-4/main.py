import analyzepeaks
import argparse
import helpers.pathfinder as pf


if __name__ == "__main__":
     ap = argparse.ArgumentParser()

     # Main argparse flags
     ap.add_argument("--folder", help="Plot files in given folder")
     ap.add_argument("--file", help="Plot given file")
     ap.add_argument("--compare",
                         help="Compare peaks between two files. Delimited by a comma")
     ap.add_argument("--tempfile", nargs="+", help="Work with tempfiles")
     ap.add_argument("--test", action='store_true', help="Test argument")

     # Optional flags
     ap.add_argument("-peaks", action='store_true',
                         help="Optional. Plot files in folder with peaks")
     ap.add_argument("-hide", action='store_true',
                         help="Optional. Hide the plots and just print text. Must also include -peaks")
     ap.add_argument("-scatter", action='store_true',
                         help="Optional. Whether or not data should be scattered.")
     ap.add_argument("-log", action="store_true", help="Optional. Whether or not y-scale should be log-scaled")
     ap.add_argument("-save", nargs='?', const="temp", help="Saves gathered data")
     ap.add_argument("-savefig", nargs='?', const="fig.pdf", help="Save figure")
     ap.add_argument("-xlim", help="X limit. Seperated by a comma, e.g., -xlim 0,100")
     ap.add_argument("-fit", action='store_true', help="Fit data")
     ap.add_argument("-labels", nargs=3, help="Title, xlabel, ylabel")

     args = ap.parse_args()
     kwargs = args.__dict__

     # Get all files in data directory
     data_files = pf.dir_file_crawler("data/")
     data_dirs = pf.dir_dir_crawler("data/")  # Get all dirs in data

     analyzer = analyzepeaks.Analyze(kwargs)

     if args.labels is not None:
          print(args.labels)
          argdict = {i:j for i,j in zip(["title", "xlabel", "ylabel"], args.labels)}
          analyzer.set_plt_titles(argdict)

     if args.folder is not None:
          folder = args.folder
          folder = pf.special_dir(folder, data_dirs)
          print(folder)

          if args.peaks:
               analyzer.plot_folder_with_gauss(
                    folder)
          else:
               analyzer.plot_folder(folder)
     elif args.file is not None:
          file = args.file
          file = pf.special_file(file, data_files)
          print(file)

          if args.peaks:
               analyzer.plot_file_with_gauss(
                    file)
          else:
               analyzer.plot_file(file)
     elif args.compare is not None:
          file1, file2 = args.compare.split(",")

          file1 = pf.special_file(file1, data_files)
          file2 = pf.special_file(file2, data_files)

          analyzer.compare_peaks(file1, file2)
     elif args.tempfile is not None:
          analyzer.tempfile(args.tempfile)
     elif args.test:
          analyzepeaks.create_calibration()

     if kwargs["save"] is not None:
          analyzer.save_data_used(kwargs["save"]+".txt")
