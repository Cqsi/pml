import numpy as np
import pandas as pd

# Test with different classifiers: NaiveBayes, DescisionTrees, RandomForestClassifier 

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
 
from prepareData import prepare_data

def train_model(path):

    df = pd.read_csv(path)

    X_train, y_train = prepare_data(df, True)

    #rfc = RandomForestClassifier(n_estimators=100)
    clf = DecisionTreeClassifier()
    fit = clf.fit(X_train, y_train)

    return fit