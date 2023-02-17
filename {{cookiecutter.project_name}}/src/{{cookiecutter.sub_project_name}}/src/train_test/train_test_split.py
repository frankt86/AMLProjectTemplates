import argparse
import pickle

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", required=True, help="data to split for train and test")
parser.add_argument("--train_size", type=float, required=True, help="proportion of data for training")
parser.add_argument("--seed", type=int, required=True, help="seed to use throughout state")
parser.add_argument("--train_data", required=True, help="data to be used for training")
parser.add_argument("--test_data", required=True, help="data to be used for testing")
args = parser.parse_args()

mlflow.start_run()

# printing script parameters
print("\n".join(f"{arg}: {val}" for arg, val in vars(args).items()))

# loading data
df_input_data = pd.read_csv(args.input_data)
print(f"Number of records: {df_input_data.shape[0]:,}")

# splitting data into train and test
df_train, df_test = train_test_split(
    df_input_data, train_size=args.train_size, random_state=args.seed
)

# passing data onto other components
df_train.to_csv(args.train_data, index=False)
df_test.to_csv(args.test_data, index=False)

mlflow.end_run()
