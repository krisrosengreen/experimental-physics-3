import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def print_underline(txt):
    print(txt)
    print("-"*len(txt))

# Get all files in the format "*deg.csv"
files = glob.glob(r"Data\Week_2\*deg.cvs")

print_underline("Found following files:")
print("\n".join(list(map(lambda x: " |-> " + x.split('\\')[-1], files))))  # Dont look at this. It's cancer for your eyes

angle_val = {}

# Load file and add data to dict called angle_val
for file in files:
    angle = int(file.split(os.sep)[-1].split('deg')[0])
    angle_val[angle] = np.loadtxt(file, skiprows=2, delimiter=';', usecols=9)

# Sort the names to be in numerical order from 0 to 90 degs
sorted_keys = sorted(angle_val.keys())

# Plot the angle on the x-axis and coincidences on the y-axis
plt.plot(sorted_keys, list([angle_val[i] for i in sorted_keys]))
plt.show()