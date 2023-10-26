from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import ClassifierMixin
from typing import Protocol, Dict, Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class PredictiveModel(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

class BinaryClassificationEvaluator:
    def __init__(self, models: Dict[str, PredictiveModel], X_test: pd.DataFrame, y_test: pd.Series, verbose: bool = True, show_plots: bool = False):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.verbose = verbose
        self.show_plots = show_plots

    def evaluate_models(self) -> None:
        for name, model in self.models.items():
            model_name = name.replace(" ", "_").lower()
            y_pred = model.predict(self.X_test)
            report_str = classification_report(self.y_test, y_pred)
            if self.verbose:
                print(f"Classification Report for {name}:")
                print(report_str)
            report_file_path = f'results/classification_report_{model_name}.txt'
            with open(report_file_path, 'w') as file:
                file.write(f"Classification Report for {name}:\n")
                file.write(report_str)
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {name}')
            plt.savefig(f'plots/confusion_matrix_{model_name}.png', bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.clf()