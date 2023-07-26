import pandas as pd
import helpers.pathfinder as pf

def load_file(filepath) -> pd.DataFrame:
    print("FILE", filepath)
    data = pd.read_csv(filepath, delimiter=";")
    data = data.rename(columns=lambda s: s.strip())

    return data

# Test above functions. Will not execute when imported.
if __name__ == "__main__":
    filename = "med_lys_og"
    load_file(pf.getfile(filename, "Data/"))
