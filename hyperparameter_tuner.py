from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os
from data_loader import DataLoader

class HyperparameterTuner:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def tune_hyperparameters(self, model: BaseEstimator, param_grid: dict = None, cv: int = 5, scoring: str = 'accuracy', n_jobs: int = None):
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
        grid_search.fit(self.X_train, self.y_train)
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        print(f"Best parameters: {best_params} with accuracy score: {best_score}")

        return best_params, best_score
        
    def tune_gda(self):
        param_grid = {
            'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        gda = QuadraticDiscriminantAnalysis()

        return self.tune_hyperparameters(model=gda, param_grid=param_grid)

    def tune_logistic_regression(self):
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
        lr = LogisticRegression(solver='liblinear', penalty='l2', fit_intercept=True, max_iter=1000, verbose=1)

        return self.tune_hyperparameters(model=lr, param_grid=param_grid)
    
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    loader = DataLoader(directory=data_dir, verbose=False)
    loader.load_and_clean_data(fresh=False)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(original_label_column='label', features=None, test_size=0.3, val_size=0.0)
    tuner = HyperparameterTuner(X_train, y_train)
    tuner.tune_logistic_regression()

if __name__ == "__main__":
    main()