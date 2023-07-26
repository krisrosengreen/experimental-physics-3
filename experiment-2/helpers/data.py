import numpy as np
import pandas as pd

class DataStructure:
    def __init__(self):
        pass

    def save(self, filename="temp.txt"):
        self.df.to_csv("temps/"+filename, index=False)

    @staticmethod
    def savedict(dict, filename="temp.txt"):
        df = pd.DataFrame.from_dict(dict)
        print(f"Columns [{', '.join(df.keys())}] have been saved to {filename}...")
        df.to_csv("temps/"+filename, index=False)

    def load(self, dict):
        self.df = pd.DataFrame.from_dict(dict)

    def loadfile(self, filename="temp.txt"):
        self.df = pd.read_csv("temps/"+filename)