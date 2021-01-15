import pandas as pd
import os

df = pd.read_csv(os.path.join("data", "train.csv"))

print(df.loc[df["Embarked"]=="S"])