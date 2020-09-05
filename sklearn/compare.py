from sklearn import linear_model

def LinearRegression_to_BayesianRidge(x_df, y_df):

    reg = linear_model.LinearRegression()
    reg.fit(x_df, y_df)

    rid = linear_model.BayesianRidge()
    rid.fit(x_df, y_df)

    scores = {
        "LinearRegression" : reg.score(x_df, y_df)
        "BayesianRidge" : rid.score(x_df, y_df)
    }

    return scores