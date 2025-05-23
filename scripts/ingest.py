import os
import pandas as pd

def main():
    # URL for the processed Cleveland dataset  

    os.makedirs("../data", exist_ok=True)

    DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data")

    # col names as per the UCL documentation
    col_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]

    # load data into DataFrame
    df = pd.read_csv(DATA_URL, names=col_names, na_values='?')

    df.to_csv('../data/heart.csv', index = False) # index = False --> don't copy the dataframe index to csv
    print('Saved data to data/heart.csv')

if __name__ == "__main__":
    main()