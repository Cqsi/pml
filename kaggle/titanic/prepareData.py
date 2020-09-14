import pandas as pd

def prepare_data(df, is_train):

    # Males = 1, females = 0
    df.Sex[df.Sex == "male"] = 1
    df.Sex[df.Sex == "female"] = 0

    # fill the NaN in age column, with the average (30)
    df.Age = df.Age.fillna(30)
    df.Fare = df.Fare.fillna(0)
    df["Family"] = df.Parch + df.SibSp
    df['Is_Alone'] = df.Family == 0
    
    columns_drop = ["Name", "SibSp", "Parch", "Ticket", "Embarked", "Cabin", "PassengerId", "Family"]

    if is_train:

        y = df.Survived
        X = df.drop(columns_drop, axis="columns").drop("Survived", axis="columns")

        return [X, y]
    
    else:

        X = df.drop(columns_drop, axis="columns")

        return X