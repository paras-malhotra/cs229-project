from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from typing import Dict, Any
import pandas as pd
import os
import joblib
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

class BinaryClassifier:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, model_dir: str ='saved_models',  verbose: bool = True) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.verbose = verbose
        self.model_dir = model_dir

    def train_models(self, retrain: bool = False) -> Dict[str, Any]:
        models = {
            'GDA': QuadraticDiscriminantAnalysis(),
            'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear', penalty='l2', fit_intercept=True, max_iter=1000, verbose=1 if self.verbose else 0),
            'Decision Tree': DecisionTreeClassifier(random_state=42, criterion='gini', splitter='best', max_depth=None, max_features=None)
        }
        for name, model in models.items():
            file_path = os.path.join(self.model_dir, f'{name.replace(" ", "_").lower()}.joblib')
            if retrain or not os.path.exists(file_path):
                # Train the model if retrain is True or the model file doesn't exist
                model.fit(self.X_train, self.y_train)
                # Save the trained model to a file
                joblib.dump(model, file_path)
                if self.verbose:
                    print(f'Trained and saved {name} model.')
            else:
                # Load the model from the file if retrain is False and the model file exists
                models[name] = joblib.load(file_path)
                if self.verbose:
                    print(f'Loaded {name} model from file.')
        return models