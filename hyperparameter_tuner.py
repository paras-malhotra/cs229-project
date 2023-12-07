import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os
from sklearn.utils import compute_class_weight
from data_loader import DataLoader
from sklearn.metrics import accuracy_score

class HyperparameterTuner:
    def __init__(self, X_train, y_train, X_val, y_val, labels_train, labels_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.labels_train = labels_train
        self.labels_val = labels_val

    def tune_hyperparameters(self, model: BaseEstimator, param_grid: dict = None, cv: int = 5, scoring: str = 'accuracy', n_jobs: int = None):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Best parameters: {best_params} with accuracy score: {best_score}")

        return best_params, best_score

    def tune_hyperparameters_holdout(self, model: BaseEstimator, param_grid: dict, multi_class: bool = False, weight_classes: bool = False):
        best_params = None
        best_score = 0
        y_val = self.labels_val if multi_class else self.y_val
        y_train = self.labels_train if multi_class else self.y_train
        class_weights = self.get_custom_class_weights()

        for params in ParameterGrid(param_grid):
            model.set_params(**params)
            if weight_classes:
                model.fit(self.X_train, y_train, sample_weight=[class_weights[label] for label in y_train])
            else:
                model.fit(self.X_train, y_train)
            preds = model.predict(self.X_val)
            score = accuracy_score(y_val, preds)

            if score > best_score:
                best_score = score
                best_params = params

            print(f"Params: {params}, Score: {score}")

        print(f"Best parameters: {best_params} with accuracy score: {best_score}")
        return best_params, best_score

    def tune_gda_binary_class(self):
        param_grid = {
            'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        gda = QuadraticDiscriminantAnalysis()

        return self.tune_hyperparameters_holdout(model=gda, param_grid=param_grid)

    def tune_logistic_regression_binary_class(self):
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        lr = LogisticRegression(solver='liblinear', penalty='l2', fit_intercept=True, max_iter=200, verbose=1)

        return self.tune_hyperparameters_holdout(model=lr, param_grid=param_grid)
    
    def tune_gda_multi_class(self):
        param_grid = {
            'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        gda = QuadraticDiscriminantAnalysis()

        return self.tune_hyperparameters_holdout(model=gda, param_grid=param_grid, multi_class=True)

    def tune_logistic_regression_multi_class(self):
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, verbose=1)

        return self.tune_hyperparameters_holdout(model=lr, param_grid=param_grid, multi_class=True, weight_classes=False)

    def get_custom_class_weights(self, max_weight=10):
        unique_classes = np.unique(self.labels_train)
        class_weights = compute_class_weight(class_weight='balanced', classes=unique_classes, y=self.labels_train)

        # Cap the weights to a maximum value
        class_weights = np.minimum(class_weights, max_weight)

        # Return a dictionary with class labels as keys
        return dict(zip(unique_classes, class_weights))

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    loader = DataLoader(directory=data_dir, verbose=False)
    loader.load_and_clean_data(fresh=True)
    X_train, X_val, X_test, y_train, y_val, y_test, labels_train, labels_val, labels_test = loader.split_data(original_label_column='label', features=None, test_size=0.1, val_size=0.1)
    tuner = HyperparameterTuner(X_train, y_train, X_val, y_val, labels_train, labels_val)
    tuner.tune_logistic_regression_multi_class()

if __name__ == "__main__":
    main()