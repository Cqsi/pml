import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from prepareData import prepare_data

def train_model(path):

    df = pd.read_csv(path)

    X_train, y_train = prepare_data(df, True)

    rfc = RandomForestClassifier(n_estimators=100)
    fit = rfc.fit(X_train, y_train)

    return fit