"""
The below file is an example of using Logistic Regression with embeddings
"""

import argparse
import json
import os
import sys
import pickle

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import mlflow

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    precision_score,
    recall_score,
)

from sklearn.linear_model import LogisticRegression

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--datastore_path", required=True, help="path to project data directory"
)
parser.add_argument("--label_col", required=True, help="column to use as label")
parser.add_argument(
    "--seed", required=False, type=int, help="seed to be used throughout the script"
)
args = parser.parse_args()

# starting mflow run
mlflow.start_run()

# logging script parameters
params = {"label_col": args.label_col, "seed": args.seed}
mlflow.log_params(params)

# loading data
features_df = pd.read_csv(args.datastore_path + "/features.csv")
ind_sec_hiearchy_df = pd.read_csv(
    args.datastore_path + "/industry_sector_hierarchy.csv"
)

# handy to have columns besides labels and features
df_train, df_test = train_test_split(
    features_df, stratify=features_df[args.label_col], random_state=args.seed
)

non_feature_cols = ["Id", "Companyid", "CompleteDescription", "Sector__c"]
feature_cols = [c for c in features_df.columns if c not in non_feature_cols]

X_train = df_train[feature_cols].to_numpy()
X_test = df_test[feature_cols].to_numpy()
y_train = df_train[args.label_col].to_numpy()
y_test = df_test[args.label_col].to_numpy()

# defining estimator for grid search
lr_clf = LogisticRegression(random_state=args.seed, max_iter=1_000)

# parameter space to search over
param_grid = {
    "C": [1, 0.75, 0.50, 0.25, 0.01],
    "solver": ["liblinear", "lbfgs"],
}

grid_search = GridSearchCV(estimator=lr_clf, param_grid=param_grid, n_jobs=-1)
grid_search.fit(X_train, y_train)

grid_search_cols = ["rank_test_score", "param_solver", "param_C", "mean_test_score"]

grid_search_results_df = pd.DataFrame(
    grid_search.cv_results_, columns=grid_search_cols
).sort_values(by="rank_test_score", ascending=True)
mlflow.log_text(
    grid_search_results_df.to_csv(index=False), "evaluation/grid_search_results.csv"
)

best_model = grid_search.best_estimator_
mlflow.sklearn.log_model(best_model, "model/")

# predicting on test set
y_pred = best_model.predict(X_test)
y_pred_prob = best_model.predict_proba(X_test)

# evaluating sector predictions on a per-industry basis
for ind in ind_sec_hiearchy_df["Industry"].unique():
    industry_sectors = ind_sec_hiearchy_df.loc[
        ind_sec_hiearchy_df["Industry"] == ind, "Sector__c"
    ].tolist()

    # relevant model classes while preserving order
    relevant_model_classes = [c for c in best_model.classes_ if c in industry_sectors]

    # keeping pred prob rows and cols where sector label is industry-appropriate
    # probabilities are normalized after narrowing pred proba scope
    row_bool_idx = df_test["Sector__c"].isin(industry_sectors)
    col_bool_idx = [c in industry_sectors for c in best_model.classes_]
    ind_specific_pred_prob = normalize(
        X=y_pred_prob[row_bool_idx][:, col_bool_idx], axis=1, norm="l1"
    )
    ind_specific_pred_prob_df = pd.DataFrame(
        data=ind_specific_pred_prob, columns=relevant_model_classes
    )

    # industry-specific predictions and true labels
    ind_specific_test = y_test[row_bool_idx]
    ind_specific_pred = ind_specific_pred_prob_df.idxmax(axis=1).to_numpy()

    # logging classification report
    classification_str = classification_report(ind_specific_test, ind_specific_pred)
    mlflow.log_text(classification_str, f"evaluation/{ind}_classification_report.txt")

    # logging confusion matrix
    conf_mtx_fig = plt.figure(figsize=(10, 10))
    conf_mtx_ax = sns.heatmap(
        confusion_matrix(ind_specific_test, ind_specific_pred),
        annot=True,
        cbar=False,
        xticklabels=relevant_model_classes,
        yticklabels=relevant_model_classes,
        cmap=plt.cm.Blues,
        linewidths=1,
    )
    conf_mtx_fig.axes.append(conf_mtx_fig)
    mlflow.log_figure(conf_mtx_fig, f"figures/{ind}_confusion_matrx.png")

    acc = accuracy_score(ind_specific_test, ind_specific_pred)
    balanced_acc = balanced_accuracy_score(ind_specific_test, ind_specific_pred)
    precision_macro = precision_score(
        ind_specific_test, ind_specific_pred, average="macro"
    )
    precision_wtd = precision_score(
        ind_specific_test, ind_specific_pred, average="weighted"
    )
    recall_macro = recall_score(ind_specific_test, ind_specific_pred, average="macro")
    recall_wtd = recall_score(ind_specific_test, ind_specific_pred, average="weighted")

    # logging scoring metrics
    metrics = {
        f"{ind}_acc": round(acc, 4),
        f"{ind}_balanced_acc": round(balanced_acc, 4),
        f"{ind}_precision_macro": round(precision_macro, 4),
        f"{ind}_precision_wtd": round(precision_wtd, 4),
        f"{ind}_recall_macro": round(recall_macro, 4),
        f"{ind}_recall_wtd": round(recall_wtd, 4),
    }

    # topk accuracy only required for industries with > 2 classes
    if len(industry_sectors) > 2:
        top_2_acc = top_k_accuracy_score(ind_specific_test, ind_specific_pred_prob)
        metrics[f"{ind}_top_2_acc"] = round(top_2_acc, 4)

    mlflow.log_metrics(metrics)

df_predictions = pd.concat(
    objs=[
        df_test[non_feature_cols].reset_index(drop=True),
        pd.Series(y_pred, name="PredictedSector"),
        pd.DataFrame(y_pred_prob, columns=best_model.classes_),
    ],
    axis=1,
)

mlflow.log_text(df_predictions.to_csv(index=False), "evaluation/predictions.csv")

mlflow.end_run()
