import helpers.pathfinder as pf

setattr(pf, "FILES", pf.dir_file_crawler("Data/"))
setattr(pf, "FOLDERS", pf.dir_dir_crawler("Data/"))