import argparse
import mlflow
from deltalake import DeltaTable

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", required=True, help="path to data directory")
parser.add_argument(
    "--processed_data", required=True, help="output path for prepared data"
)
args = parser.parse_args()

mlflow.start_run()

# printing script parameters
print("\n".join(f"{arg}: {val}" for arg, val in vars(args).items()))

# paths to data
data_stage_path = (
    args.input_path + "Enter Path Here "
)  # i.e. "/stage/ib/salesforce/prod/current/"


# columns to read
cols = ["col1", "col2", "col3"]


df_object = DeltaTable(data_stage_path + "Object_Name").to_pandas(columns=cols)

print(f"Object Records: {df_object.shape[0]:,}")


"""
Dataframe Manipulation goes here
"""

# pass on to next component
df_object.to_csv(args.processed_data, index=False)

mlflow.end_run()
