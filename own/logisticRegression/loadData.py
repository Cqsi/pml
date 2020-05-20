import pandas as pd

def loadData(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df
