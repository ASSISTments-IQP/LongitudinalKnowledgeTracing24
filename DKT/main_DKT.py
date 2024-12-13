import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DKT_pt import DKT


def main():
    file_path = '../Data/samples/21-22/sample1.csv'
    test_file_path = '../Data/samples/22-23/sample1.csv'
    #'23-24-problem_logs.csv'

    train = pd.read_csv(file_path)
    test = pd.read_csv(test_file_path)

    model = DKT(feature_col='skill_id')

    model.fit(train, num_epochs=3)

    model.evaluate(test)


if __name__ == "__main__":
    main()
