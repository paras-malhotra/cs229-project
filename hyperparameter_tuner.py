from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import os
from data_loader import DataLoader

class HyperparameterTuner:
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def tune_hyperparameters(self):
        param_grid = {
            'reg_param': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        }
        gda = QuadraticDiscriminantAnalysis()
        grid_search = GridSearchCV(gda, param_grid, cv=5, scoring='accuracy', n_jobs=None)
        grid_search.fit(self.X_train, self.y_train)
        best_reg_param = grid_search.best_params_['reg_param']
        best_score = grid_search.best_score_
        print(f"Best regularization parameter for GDA: {best_reg_param} with accuracy score: {best_score}")

        return best_reg_param, best_score
    
def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    loader = DataLoader(directory=data_dir, verbose=False)
    loader.load_and_clean_data(fresh=False)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(original_label_column='label', features=None, test_size=0.3, val_size=0.0)
    tuner = HyperparameterTuner(X_train, y_train)
    tuner.tune_hyperparameters()

if __name__ == "__main__":
    main()