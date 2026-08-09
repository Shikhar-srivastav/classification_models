import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from constants.common import target_cols

def evaluate_model(y_test, predictions, probabilities):
    model_metrics = {
        "Accuracy": metrics.accuracy_score(y_test, predictions),
        "AUC": metrics.roc_auc_score(y_test, probabilities, multi_class="ovr"),
        "Precision": metrics.precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": metrics.recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1": metrics.f1_score(y_test, predictions, average="weighted", zero_division=0),
        "MCC": metrics.matthews_corrcoef(y_test, predictions),
    }

    return model_metrics

def create_confusion_matrix(y_test, predictions):
    class_labels = target_cols
    cm = metrics.confusion_matrix(y_test, predictions)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, 
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels, 
        yticklabels=class_labels,
        ax=ax
    )
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")

    return fig