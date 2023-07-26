import helpers.pathfinder as pf
import helpers.mcaloader as mca

setattr(pf, "FILES", pf.dir_file_crawler("Data/"))
setattr(pf, "FOLDERS", pf.dir_dir_crawler("Data/"))

def quick_load_mca(filename):
    """
    Shortcut to getting file using pathfinder and loading using load_mca function in mcaloader file
    """

    filepath = pf.getfile_in_L(filename, pf.FILES)
    return mca.load_mca(filepath)

setattr(mca, "quick_load_mca", quick_load_mca)