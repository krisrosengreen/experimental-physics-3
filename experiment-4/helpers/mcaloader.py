import numpy as np

def load_mca(filename):
    with open(filename, "r") as file:
        lines = file.readlines()

        start = None
        end = None

        for i in range(len(lines)):
            if lines[i] == "<<DATA>>\n":
                start = i
            elif lines[i] == "<<END>>\n":
                end = i
                break

        data_lines = lines[start + 1:end]
        processed_lines = []

        for i in data_lines:
            processed_lines.append(int(i.strip()))

        return np.array(processed_lines)
