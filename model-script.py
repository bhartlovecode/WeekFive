#!/usr/bin/env python3

from pycaret.classification import predict_model, load_model
import pandas as pd
import pickle as pk


def load_file(modelFile):
    with open(modelFile, 'rb') as f:
        return pk.load(f)

class Model():
    def __init__(self, modelFile):
        self.model = load_file(modelFile)
    
    def test_model(self, df):
        predictions = predict_model(self.model, data=df)
        predictions.rename({'Label': 'Churn_Predicition'}, axis=1, inplace=True)
        predictions['Churn_Predicition'].replace({1: 'Churn', 0: 'No Churn'},
                                                inplace=True)
        return predictions['Churn_Predicition']


def load_data(filepath):
    """
    Loads churn data into a DataFrame from a string filepath.
    """
    df = pd.read_csv(filepath)
    return df

if __name__ == "__main__":
    gbc = Model("GBCmodel.pkl")
    df = load_data("new_churn_data.csv")
    predictions = gbc.test_model(df)
    print(f"Predictions: \n{predictions}")
