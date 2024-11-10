import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DKT_Model import DKT

def main():
   file_path = '23-24-problem_logs.csv'
   #'../Data/non_skill_builder_data_new.csv'
   

   df = pd.read_csv(file_path)
   df = df.sample(frac=0.10, random_state=69)
   train, test= train_test_split(df, test_size=0.2, random_state=200)

   model = DKT(50, 64, verbose=True)

   model.fit(train)

   # model.evaluate(test)

if __name__ == "__main__":
    main()