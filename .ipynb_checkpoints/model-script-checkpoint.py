#!/usr/bin/env python3

from imp import load_module
import pandas as pd
import pickle as pk

def load_model(modelFile):
        return pk.load(modelFile)

class Model():

    def __init__(self, modelFile):
        self.model = load_model(modelFile)
    
    def test_model(df):
        print(df)

if __name__ == "__main__":
    lrm = Model()
