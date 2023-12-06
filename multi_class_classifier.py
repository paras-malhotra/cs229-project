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

class MultiClassClassifier:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series, model_dir: str = 'saved_models', verbose: bool = True) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.verbose = verbose
        self.model_dir = model_dir

    def load_or_train_models(self, retrain: bool = False) -> Dict[str, Any]:
        models = {
            'GDA': QuadraticDiscriminantAnalysis(),
            'Regularized GDA': QuadraticDiscriminantAnalysis(reg_param=0),
            'Logistic Regression': LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', C=100, max_iter=1000, verbose=1 if self.verbose else 0),
            'Decision Tree': DecisionTreeClassifier(random_state=42, criterion='gini', splitter='best', max_depth=None, max_features=None),
            # 'Random Forest': RandomForestClassifier(random_state=42, criterion='gini', max_depth=None, max_features='sqrt', n_jobs=-1, verbose=3 if self.verbose else 0),
            # 'SVM': SVC(random_state=42, kernel='rbf', decision_function_shape='ovr', verbose=3 if self.verbose else 0),
            'Neural Network': MLPClassifier(random_state=42, hidden_layer_sizes=(100,), batch_size=512, activation='relu', solver='adam', alpha=0.0001, max_iter=1000, learning_rate_init=0.001, early_stopping=True, verbose=1 if self.verbose else 0),
        }
        for name, model in models.items():
            file_path = os.path.join(self.model_dir, f'multi_{name.replace(" ", "_").lower()}.joblib')
            if retrain or not os.path.exists(file_path):
                model.fit(self.X_train, self.y_train)
                joblib.dump(model, file_path)
                if self.verbose:
                    print(f'Trained and saved multi-class {name} model.')
            else:
                models[name] = joblib.load(file_path)
                if self.verbose:
                    print(f'Loaded multi-class {name} model from file.')
        return models