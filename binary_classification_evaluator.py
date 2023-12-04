from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from typing import Protocol, Dict, Any
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class PredictiveModel(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

class BinaryClassificationEvaluator:
    def __init__(self, models: Dict[str, PredictiveModel], X_test: pd.DataFrame, y_test: pd.Series, labels_test: pd.Series, scaler: StandardScaler, verbose: bool = True, show_plots: bool = False):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        self.labels_test = labels_test
        self.scalar = scaler
        self.verbose = verbose
        self.show_plots = show_plots

    def evaluate_models(self) -> None:
        for name, model in self.models.items():
            model_name = name.replace(" ", "_").lower()
            y_pred = model.predict(self.X_test)
            report_str = classification_report(self.y_test, y_pred, digits=4)
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

    def calculate_attack_identification_rate_by_label(self) -> pd.DataFrame:
        unique_labels = self.labels_test.unique()
        rates = []
        for name, model in self.models.items():
            for label in unique_labels:
                if label == 'BENIGN':
                    continue
                y_pred = model.predict(self.X_test[self.labels_test == label])
                correct_identifications = np.sum(y_pred == 1)
                total_attacks = len(y_pred)
                rate = (correct_identifications / total_attacks) * 100 if total_attacks > 0 else 0
                rates.append({'Label': label, 'Identification Rate (%)': rate})
            rate_df = pd.DataFrame(rates)
            rate_df.to_csv(f'results/attack_identification_rates_{name}.csv', index=False)
        return rate_df

    def print_top_k_high_confidence_errors(self, k: int = 100) -> None:
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                probas = model.predict_proba(self.X_test)
                y_pred = model.predict(self.X_test)
                error_indices = np.where(y_pred != self.y_test)[0]
                fp_error_indices = error_indices[probas[error_indices, 1].argsort()[-k:]]
                fn_error_indices = error_indices[probas[error_indices, 0].argsort()[-k:]]

                fp_df = pd.DataFrame(self.scalar.inverse_transform(self.X_test.iloc[fp_error_indices]), columns=self.X_test.columns)
                fp_df['Predicted Label'] = 1
                fp_df['Original Label'] = self.labels_test.iloc[fp_error_indices].values
                fp_df['Probability'] = probas[fp_error_indices, 1]

                fn_df = pd.DataFrame(self.scalar.inverse_transform(self.X_test.iloc[fn_error_indices]), columns=self.X_test.columns)
                fn_df['Predicted Label'] = 0
                fn_df['Original Label'] = self.labels_test.iloc[fn_error_indices].values
                fn_df['Probability'] = probas[fn_error_indices, 0]

                # print(f'Top {k} high confidence false positives for {name}:')
                # print(fp_df)
                fp_df.to_csv(f'results/top_{k}_fp_{name}.csv', index=False)

                # print(f'Top {k} high confidence false negatives for {name}:')
                # print(fn_df)
                fn_df.to_csv(f'results/top_{k}_fn_{name}.csv', index=False)
            else:
                print(f'{name} does not support predict_proba()')