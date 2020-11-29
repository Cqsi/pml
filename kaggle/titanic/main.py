import numpy as np
import pandas as pd
from train import train_model
from prepareData import prepare_data

df_test = pd.read_csv("data\\test.csv")
X_test = prepare_data(df_test, False)

fit = train_model("data\\train.csv")
y_pred = fit.predict(X_test)

# PREDICTION IS READY

passenger_ids = df_test.PassengerId.values

pd.DataFrame(np.column_stack((passenger_ids, y_pred))).to_csv("file.csv", header=["PassengerId", "Survived"],index=None)


# TESTING WITH OWN VALUES
# print(fit.predict([np.array([1, 1, 48])]))