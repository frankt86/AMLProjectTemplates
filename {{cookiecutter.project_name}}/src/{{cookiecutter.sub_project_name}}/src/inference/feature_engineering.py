"""
The below file is an example of creating embedding for use in Logistic Regression
"""

import argparse
import os

import mlflow
import pandas as pd
from deltalake import DeltaTable
from sentence_transformers import SentenceTransformer


def read_dir(dir: str, target_ext: str, **kwargs) -> pd.DataFrame:
    """
    Reads in a directory of like-extension files into a dataframe

    Parameters:
    -----------
        dir (str): directory containing the files
        target_ext: (str): extension of the files to read in
        **kwargs: keywords arguments accepted by pd.read_csv

    Returns:
    --------
        complete_df (pd.DataFrame): dataframe of all files
    """

    directory_contents = os.listdir(dir)
    target_files = [fname for fname in directory_contents if fname.endswith(target_ext)]
    complete_df = pd.concat(
        [pd.read_csv(dir + f"/{fname}", **kwargs) for fname in target_files]
    )

    return complete_df


parser = argparse.ArgumentParser()
parser.add_argument("--datastore_path", help="input datastore path")
parser.add_argument("--output_data", help="feature engineered data output path")
args = parser.parse_args()

# start logging
mlflow.start_run()

# printing script parameters
print("\n".join(f"{arg}: {val}" for arg, val in vars(args).items()))

pb_stage_path = args.datastore_path + "/stage/ib/pitchbook/prod/current/"
pb_cdm_path = args.datastore_path + "/application/ib/CDM/IB_Pitchbook_Model/Other/"

pb_company_cols = [
    "Companyid",
    "Companyname",
    "Description",
    "Keywords",
    "Primaryindustrysector",
    "Primaryindustrygroup",
    "Primaryindustrycode",
]

# reading in pitchbook companies, as well as their classifications
df_pb_company = DeltaTable(pb_stage_path + "Company").to_pandas(columns=pb_company_cols)
df_classified_cos = read_dir(
    dir=pb_cdm_path + "PB_Industry_Sector_Classification.csv",
    target_ext=".csv",
    sep="\t",
)

print(f"PB company records: {df_pb_company.shape[0]:,}")
print(f"PB classifed company records: {df_classified_cos.shape[0]:,}")

# conditions to filter on
has_pb_descr_mask = df_pb_company["Description"].notna()
has_pb_id_mask = df_pb_company["Companyid"].notna()
not_classified_mask = ~df_pb_company["Companyid"].isin(df_classified_cos["Companyid"])
to_exclude_cond = df_pb_company["Companyid"] == "112228-39"

# filtering for companies to classify
df_to_classify = (
    df_pb_company.loc[
        has_pb_descr_mask
        & has_pb_id_mask
        # & not_classified_mask
        & ~to_exclude_cond
    ]
    .sample(5)
    .copy()
)

print(f"PB companies to classify: {df_to_classify.shape[0]:,}")

keyword_columns = [
    "Keywords",
    "Primaryindustrysector",
    "Primaryindustrygroup",
    "Primaryindustrycode",
]

# creating a complete description column
for col in keyword_columns:
    df_to_classify[col] = df_to_classify[col].fillna("").str.lower()

# concatenating description columns into single description column
df_to_classify["CompleteDescription"] = (
    df_to_classify["Description"]
    + ", "
    + df_to_classify["Keywords"]
    + ", "
    + df_to_classify["Primaryindustrysector"]
    + ", "
    + df_to_classify["Primaryindustrygroup"]
    + ", "
    + df_to_classify["Primaryindustrycode"]
)

# creating embeddings
model = SentenceTransformer("all-mpnet-base-v2")
df_embeddings = pd.DataFrame(
    data=model.encode(df_to_classify["CompleteDescription"].tolist()),
    index=df_to_classify.index,
)

# data sent to inferencing component will have
# company id, complete description, and the embedding columns
df_to_classify = df_to_classify[["Companyid", "CompleteDescription"]]
df_to_classify = pd.concat([df_to_classify, df_embeddings], axis=1)

# passing onto next component
df_to_classify.to_csv(args.output_data, index=False)

mlflow.end_run()
