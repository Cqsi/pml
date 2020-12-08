from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import pandas as pd
import numpy as np

X_full = pd.read_csv("data\\train.csv", index_col="Id")
X_test_full = pd.read_csv("data\\test.csv", index_col="Id")

y = X_full["SalePrice"]
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
subX_test = X_test_full[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=0)

#print(X.head())


# OPTIONAL CODE, this is used to find the best estimator amount. I put this here so that you remember how to pick the best amount; is this how to do it?

# Define the models
#model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
#model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
rfr = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0) # model_3
#model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
#model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

#models = [model_1, model_2, model_3, model_4, model_5]


# Function for comparing different models
#def score_model(model, X_t=X_train, X_v=X_test, y_t=y_train, y_v=y_test):
#    model.fit(X_t, y_t)
#    preds = model.predict(X_v)
#    return mean_absolute_error(y_v, preds)

#for i in range(0, len(models)):
#    mae = score_model(models[i])
#    print("Model %d MAE: %d" % (i+1, mae))

# best model = model 3

model = rfr.fit(X_train, y_train)
model.fit(X,y)

# check the model on existing test data (from the kaggle train dataset)
#preds = model.predict(X_test)
#print(r2_score(y_test, preds))

preds = model.predict(subX_test)

# output the data
output = pd.DataFrame({'Id': subX_test.index,
                       'SalePrice': preds})
output.to_csv('data\\submission.csv', index=False)
print("Prediction is ready. Check output file.")