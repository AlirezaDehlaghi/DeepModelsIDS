# This is helper function Version 2

import random
from datetime import datetime
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


# Set random seeds for reproducibility
def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)


class Scores:
    accuracy = 'Accuracy'
    precision = 'Precision'
    recall = 'Recall'
    f1 = 'Macro F1 Score'
    weighted_f1 = 'Weighted F1 Score'
    computed_auc = 'AUC'


def compute_score(y_true, y_pred, y_pred_prob=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,  average='macro', zero_division=np.nan)
    recall = recall_score(y_true, y_pred,  average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    if y_pred_prob is not None:
        computed_auc = roc_auc_score(y_true, y_pred_prob, average='macro', multi_class='ovo')
    else:
        computed_auc = 'Not Computed!'
    cm = confusion_matrix(y_true, y_pred)

    return {Scores.accuracy: accuracy,
            Scores.precision: precision,
            Scores.recall: recall,
            Scores.f1: f1,
            Scores.weighted_f1: weighted_f1,
            Scores.computed_auc: computed_auc}, cm


# create sequences for variable window sizes
def create_sequences(features, labels, window_size, step_size=1):
    x, y = [], []
    for i in range(0, len(features) - window_size + 1, step_size):
        x.append(features[i:(i + window_size)])
        y.append(labels[i + window_size - 1])
    return np.array(x), np.array(y)


# ANSI escape sequences for colors
class Bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Report:
    @staticmethod
    def green(text):
        Report._make_color(text, Bcolors.GREEN)

    @staticmethod
    def blue(text):
        Report._make_color(text, Bcolors.BLUE)

    @staticmethod
    def cyan(text):
        Report._make_color(text, Bcolors.CYAN)

    @staticmethod
    def debug(text):
        Report._make_color(text, Bcolors.FAIL)

    @staticmethod
    def warn(text):
        Report._make_color(text, Bcolors.WARNING)

    @staticmethod
    def normal(text):
        Report._base_report(text)

    @staticmethod
    def _make_color(text, color):
        Report._base_report(f"{color}{text}{Bcolors.END}")

    @staticmethod
    def _base_report(text):
        now = datetime.now()
        formatted_time = now.strftime("%H:%M:%S")
        print(f"[{formatted_time}] {text}")

    @staticmethod
    def test_class_function():
        print("test_class_function")




def test_function():
    print("test_function9")

