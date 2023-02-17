import argparse

import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_data", required=True, help="path of data to embed")
parser.add_argument("--output_data", required=True, help="data with engineered features")
args = parser.parse_args()

mlflow.start_run()

# printing script parameters
print("\n".join(f"{arg}: {val}" for arg, val in vars(args).items()))

# loading data
df = pd.read_csv(args.input_data)
print(f"Shape of data: {df.shape}")

keyword_columns = [
    "Keywords",
    "Primaryindustrysector",
    "Primaryindustrygroup",
    "Primaryindustrycode",
]

# keyword columns are pre-processed before submitted 
for col in keyword_columns:
    df[col] = df[col].fillna("").str.lower()

# concatenating text columns into single description column
all_text_columns = ["Description"] + keyword_columns
df["CompleteDescription"] = df[all_text_columns].agg(", ".join, axis=1)

# creating embeddings
model = SentenceTransformer("all-mpnet-base-v2")
df_embeddings = pd.DataFrame(
    data=model.encode(df["CompleteDescription"].tolist())
)

# combining data
df = df[["Id", "PitchBook_ID__c", "CompleteDescription", "Industry"]]
df_combined = pd.concat([df, df_embeddings], axis=1)
print(f"Shape of data (post-engineering): {df_combined.shape}")

# passing onto next component
df_combined.to_csv(args.output_data, index=False)

mlflow.end_run()
