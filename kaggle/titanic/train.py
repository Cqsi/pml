import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from prepareData import prepare_data

def train_model(path):

    df = pd.read_csv(path)

    X_train, y_train = prepare_data(df, True)

    gnb = GaussianNB()
    fit = gnb.fit(X_train, y_train)

    return fit