import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import pandas as pd
import os
import joblib
from typing import Dict, Any
from sklearn.utils import compute_class_weight
from sklearn.metrics import classification_report

class MultiClassClassifier:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, model_dir: str = 'saved_models', verbose: bool = True) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.verbose = verbose
        self.model_dir = model_dir

    def load_or_train_models(self, retrain: bool = False, weight_classes: bool = False) -> Dict[str, Any]:
        models = {
            'GDA': QuadraticDiscriminantAnalysis(),
            'Regularized GDA': QuadraticDiscriminantAnalysis(reg_param=0),
            'Logistic Regression': LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', C=100, max_iter=1000, verbose=1 if self.verbose else 0),
            'Decision Tree': DecisionTreeClassifier(random_state=42, criterion='gini', splitter='best', max_depth=None, max_features=None),
            # 'Random Forest': RandomForestClassifier(random_state=42, criterion='gini', max_depth=None, max_features='sqrt', n_jobs=-1, verbose=3 if self.verbose else 0),
            # 'SVM': SVC(random_state=42, kernel='rbf', decision_function_shape='ovr', verbose=3 if self.verbose else 0),
            'Neural Network': MLPClassifier(random_state=42, hidden_layer_sizes=(100,), batch_size=512, activation='relu', solver='adam', alpha=0.0001, max_iter=1000, learning_rate_init=0.001, early_stopping=True, verbose=1 if self.verbose else 0),
        }

        class_weights = self.get_custom_class_weights()

        for name, model in models.items():
            file_path = os.path.join(self.model_dir, f'multi_{name.replace(" ", "_").lower()}.joblib')
            if retrain or not os.path.exists(file_path):
                sample_weights = [class_weights[label] for label in self.y_train]
                if not weight_classes or name in ['GDA', 'Regularized GDA', 'Neural Network']:
                    model.fit(self.X_train, self.y_train)
                else:
                    model.fit(self.X_train, self.y_train, sample_weights)
                joblib.dump(model, file_path)
                if self.verbose:
                    print(f'Trained and saved multi-class {name} model.')
            else:
                model = joblib.load(file_path)
                models[name] = model
                if self.verbose:
                    print(f'Loaded multi-class {name} model from file.')

            if self.verbose:
                y_train_pred = model.predict(self.X_train)
                train_report = classification_report(self.y_train, y_train_pred, digits=4, zero_division=0)
                print(f"Training Classification Report for {name}:")
                print(train_report)

                # Save the training report to a file
                train_report_file_path = os.path.join('results', f'training_classification_report_{name.replace(" ", "_").lower()}.txt')
                with open(train_report_file_path, 'w') as file:
                    file.write(f"Training Classification Report for {name}:\n")
                    file.write(train_report)
        return models

    def get_custom_class_weights(self, max_weight=10):
        unique_classes = np.unique(self.y_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=self.y_train)

        # Cap the weights to a maximum value
        class_weights = np.minimum(class_weights, max_weight)

        # Return a dictionary with class labels as keys
        return dict(zip(unique_classes, class_weights))