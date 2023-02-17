import argparse

import pandas as pd
import mlflow
from sklearn.linear_model import LogisticRegression

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train_data", required=True, help="data to be used for modeling")
parser.add_argument("--C", required=True, type=float, help="regularization parameter")
parser.add_argument("--solver", required=True, help="solver to use")
parser.add_argument("--max_iter", required=True, type=int, help="max iterations for convergence")
parser.add_argument("--seed", required=False, type=int, help="seed to use throughout the script")
parser.add_argument("--model_dir", required=False, help="trained model directory")
args = parser.parse_args()

mlflow.start_run()

# loading data
df_train = pd.read_csv(args.train_data)

# names of the embeddings columns
predictors = [
    col
    for col in df_train.columns
    if col not in ["Id", "PitchBook_ID__c", "CompleteDescription", "Industry"]
]

X_train = df_train[predictors].to_numpy()
y_train = df_train["Industry"].to_numpy()

# training the model
lr_clf = LogisticRegression(
    C=args.C, solver=args.solver, max_iter=args.max_iter, random_state=args.seed
)
lr_clf.fit(X_train, y_train)

# saving the model as an artifact and as file for next component
mlflow.sklearn.save_model(sk_model=lr_clf, path=args.model_dir + "/trained_model")

mlflow.end_run()
