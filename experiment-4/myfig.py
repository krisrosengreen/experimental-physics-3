import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import helpers.cosmic_loader as cl

if __name__ == "__main__":
    df = cl.load_file("Data/Week_2/weekendmaaling.txt")

    A = df['A'].to_numpy()

    print(df)
