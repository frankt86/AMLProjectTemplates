"""
The below file is an example of creating embedding for use in Logistic Regression
"""

import argparse
import re

import pandas as pd

from sentence_transformers import SentenceTransformer

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--datastore_path", required=True, help="path to description data")
args = parser.parse_args()

# print argument parameters to terminal
print("\n".join(f"{arg}: {val}" for arg, val in vars(args).items()))

# loading data
df = pd.read_csv(args.datastore_path + "/processed_data.csv")
print(f"Number of records: {df.shape[0]:,}")

# industry needed to predict sector
onehot_industry_df = pd.get_dummies(df["Industry"])
onehot_industry_df.columns = [
    re.sub(r"[^A-Za-z0-9]", "", c) for c in onehot_industry_df.columns
]

keyword_columns = [
    "Keywords",
    "Primaryindustrysector",
    "Primaryindustrygroup",
    "Primaryindustrycode",
]

for col in keyword_columns:
    df[col] = df[col].fillna("").str.lower()

# concatenating text columns into single description column
all_text_columns = ["Description"] + keyword_columns
df["CompleteDescription"] = df[all_text_columns].agg(", ".join, axis=1)

# bert embeddings will also be features
model = SentenceTransformer("all-mpnet-base-v2")
embeddings = model.encode(df["CompleteDescription"].tolist())

# creating feature dataframe from industry and description encodings
features_df = pd.concat(
    objs=[
        df[["Id", "Companyid", "CompleteDescription", "Sector__c"]],
        onehot_industry_df,
        pd.DataFrame(embeddings),
    ],
    axis=1,
)

# saving embeddings
features_df.to_csv("outputs/features.csv", index=False)
