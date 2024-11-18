import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DKT_pt import DKT


def main():
    file_path = '../Data/samples/21-22/sample1.csv'
    #'23-24-problem_logs.csv'

    df = pd.read_csv(file_path)
    df = df.sample(frac=0.05, random_state=69)
    print(df.groupby(by=['user_xid']).size().mean())
    train, test = train_test_split(df, test_size=0.2, random_state=200)

    model = DKT(feature_col='old_problem_id')

    model.fit(train)

    model.eval(test)


if __name__ == "__main__":
    main()
