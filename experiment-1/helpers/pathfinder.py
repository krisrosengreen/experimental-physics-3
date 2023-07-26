from genericpath import isdir, isfile
import os
from difflib import SequenceMatcher

def contains_only_files(path):
    for file in os.listdir(path):
        if os.path.isdir(file):
            return False
    return True


def matches_name(name1, name2, threshold = 0.90):
    match = SequenceMatcher(a=name1.lower(), b=name2.lower()).ratio()
    return match > threshold, match

#
# FILES
#

def dir_file_crawler(root_path):
    files = []

    for name in os.listdir(root_path):
        path_name = root_path + name
        if os.path.isdir(path_name):
            L = dir_file_crawler(path_name+"/")
            files = files + L
        else:
            files.append(path_name)
    return files


def isfile_in_L(filename, L):
    for i in L:
        name = i.split("/")[-1].split(".")[0]
        if matches_name(name, filename)[0]:
            return True


def getfile_in_L(filename, L):
    best_name = ""
    best_match = 0
    for i in L:
        name = i.split("/")[-1].split(".")[0]
        match_name, match_score = matches_name(name, filename)
        if match_score > best_match:
            best_name = i
            best_match = match_score

    return best_name


def getfile(filename, foldername):
    L = dir_file_crawler(foldername)
    return getfile_in_L(filename, L)


def hasfile(filename, foldername):
    L = dir_file_crawler(foldername)
    return isfile_in_L(filename, foldername)


def special_file(name, data_files):
    if '@' in name:
        name.replace('@', '')

        assert isfile_in_L(
            name, data_files), f"File, {name}, could not be found!"

        name = getfile_in_L(name, data_files)
    return name

#
# FOLDERS
#


def dir_dir_crawler(root_path):
    folders = []

    for name in os.listdir(root_path):
        path_name = root_path + name

        if os.path.isdir(path_name):
            if contains_only_files(path_name):
                folders.append(path_name+"/")
            else:
                L = dir_dir_crawler(path_name)
                folders = folders + L
    return folders


def special_dir(name, data_files):
    if '@' in name:
        name.replace('@', '')

        assert isdir_in_L(
            name, data_files), f"File, {name},could not be found!"

        name = getdir_in_L(name, data_files)
    return name


def isdir_in_L(dirname, L):
    for i in L:
        name = i.split("/")[-2]
        if matches_name(name, dirname)[0]:
            return True


def getdir_in_L(dirname, L):
    for i in L:
        name = i.split("/")[-2]
        if matches_name(name, dirname)[0]:
            return i
