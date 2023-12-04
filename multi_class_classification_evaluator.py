from sklearn.metrics import classification_report, confusion_matrix
from typing import Protocol, Dict, Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

class PredictiveModel(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

class MultiClassClassificationEvaluator:
    def __init__(self, models: Dict[str, PredictiveModel], X_test: pd.DataFrame, y_test: pd.Series, scaler: StandardScaler, verbose: bool = True, show_plots: bool = False):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler
        self.verbose = verbose
        self.show_plots = show_plots

    def evaluate_models(self) -> None:
        for name, model in self.models.items():
            model_name = name.replace(" ", "_").lower()
            y_pred = model.predict(self.X_test)
            report_str = classification_report(self.y_test, y_pred, digits=4, zero_division=0)
            if self.verbose:
                print(f"Classification Report for {name}:")
                print(report_str)
            report_file_path = f'results/multiclass_classification_report_{model_name}.txt'
            with open(report_file_path, 'w') as file:
                file.write(f"Classification Report for {name}:\n")
                file.write(report_str)

            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix for {name}')
            plt.savefig(f'plots/multiclass_confusion_matrix_{model_name}.png', bbox_inches='tight')
            if self.show_plots:
                plt.show()
            plt.clf()

    def print_top_k_errors(self, k: int = 100) -> None:
        for name, model in self.models.items():
            model_name = name.replace(" ", "_").lower()
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(self.X_test)
                y_pred = model.predict(self.X_test)
                error_indices = np.where(y_pred != self.y_test)[0]
                top_error_indices = np.argsort(-np.max(probas[error_indices], axis=1))[:k]

                error_df = pd.DataFrame(self.scaler.inverse_transform(self.X_test.iloc[top_error_indices]), columns=self.X_test.columns)
                error_df['Predicted Label'] = y_pred[top_error_indices]
                error_df['Original Label'] = self.y_test.iloc[top_error_indices].values
                error_df['Max Probability'] = np.max(probas[top_error_indices], axis=1)

                error_df.to_csv(f'results/top_{k}_errors_{model_name}.csv', index=False)
            else:
                print(f'{name} does not support predict_proba()')
