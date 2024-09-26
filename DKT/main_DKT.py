import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from DKT_Model import DKT
import logging

def main():
   logging.info("Starting main execution")
   file_path = '23-24-problem_logs.csv'

   df = pd.read_csv(file_path)
   train, test= train_test_split(df, test_size=0.2, random_state=200)

   model = DKT()

   logging.info("Beginning model training")
   model.fit(train)
   
   logging.info("Model training finished, beginning evaluation")
   model.eval(test)

if __name__ == "__main__":
    main()