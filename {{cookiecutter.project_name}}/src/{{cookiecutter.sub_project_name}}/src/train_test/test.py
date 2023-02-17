import argparse

import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    precision_score,
    recall_score
)

from utils.secondary_class import accuracy_industry_tradeoff, thresholded_pred

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, help="model to evaluate")
parser.add_argument("--test_data", required=True, help="data to be used for testing model")
args = parser.parse_args()

mlflow.start_run()

# loading data and model
lr_clf = mlflow.sklearn.load_model(args.model_dir + "/trained_model")
df_test = pd.read_csv(args.test_data)

# names of the embeddings columns
predictors = [
    col
    for col in df_test.columns
    if col not in ["Id", "PitchBook_ID__c", "CompleteDescription", "Industry"]
]

X_test = df_test[predictors].to_numpy()
y_test = df_test["Industry"].to_numpy()

# training the model
y_pred = lr_clf.predict(X_test)
y_pred_prob = lr_clf.predict_proba(X_test)

# logging classification report
classification_str = classification_report(y_test, y_pred)
mlflow.log_text(classification_str, "outputs/classification_report.txt")

# logging various metrics
test_metrics = {
    "acc": round(accuracy_score(y_test, y_pred), 4),
    "balanced_acc": round(balanced_accuracy_score(y_test, y_pred), 4),
    "top_2_acc": round(top_k_accuracy_score(y_test, y_pred_prob), 4),
    "precision_macro": round(precision_score(y_test, y_pred, average="macro"), 4),
    "precision_weighted": round(precision_score(y_test, y_pred, average="weighted"), 4),
    "recall_macro": round(recall_score(y_test, y_pred, average="macro"), 4),
    "recall_weighted": round(recall_score(y_test, y_pred, average="weighted"), 4)
}
mlflow.log_metrics(test_metrics)

# logging confusion matrix
conf_mtx_fig = plt.figure(figsize=(10, 10))
conf_mtx_ax = sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    cbar=False,
    xticklabels=lr_clf.classes_,
    yticklabels=lr_clf.classes_,
    cmap=plt.cm.Blues,
    linewidths=1
)
conf_mtx_fig.axes.append(conf_mtx_fig)
mlflow.log_figure(conf_mtx_fig, "figures/confusion_matrix.png")

# logging trade-off between accuracy and number of classes
trade_off_fig, prob_thresh, prop_single, acc_adj = accuracy_industry_tradeoff(
    model=lr_clf,
    y_true=y_test,
    y_pred_proba=y_pred_prob,
    probabilities=np.linspace(0.01, 0.50, 500)
)
mlflow.log_figure(trade_off_fig, "figures/acc_ind_tradeoff.png")

trade_off_metrics = {
    "primary_industry_threshold": prob_thresh,
    "prop_single": prop_single,
    "adjusted_accuracy": acc_adj,
}
mlflow.log_metrics(trade_off_metrics)

# recalculating classification report using new thresholded predictions
y_pred_new, _ = thresholded_pred(
    model=lr_clf,
    y_true=y_test,
    y_pred_proba=y_pred_prob,
    value=prob_thresh
)
classification_str = classification_report(y_test, y_pred_new)
mlflow.log_text(classification_str, "outputs/classification_report_thresholded.txt")

# logging confusion matrix
conf_mtx_fig = plt.figure(figsize=(10, 10))
conf_mtx_ax = sns.heatmap(
    confusion_matrix(y_test, y_pred_new),
    annot=True,
    cbar=False,
    xticklabels=lr_clf.classes_,
    yticklabels=lr_clf.classes_,
    cmap=plt.cm.Blues,
    linewidths=1
)
conf_mtx_fig.axes.append(conf_mtx_fig)
mlflow.log_figure(conf_mtx_fig, "figures/confusion_matrix_thresholded.png")

mlflow.end_run()
