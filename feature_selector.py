from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os

class FeatureSelector:
    def __init__(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True, show_plots: bool = False):
        self.X = X
        self.y = y
        self.verbose = verbose
        self.show_plots = show_plots

    def model_based_selection(self, fresh_analysis: bool = False) -> None:
        cache_file_path = os.path.join("results", "feature_importances.pkl")

        if not fresh_analysis and os.path.exists(cache_file_path):
            # Load cached results if fresh analysis is not required and cache file exists
            with open(cache_file_path, "rb") as f:
                ordered_importances, ordered_feature_names = pickle.load(f)
        else:
            # Perform fresh analysis
            model = RandomForestClassifier(random_state=42, n_jobs=-1, verbose=3 if self.verbose else 0)
            model.fit(self.X, self.y)
            importances = model.feature_importances_
            # Order feature importances and corresponding feature names in decreasing order
            indices = importances.argsort()[::-1]
            ordered_importances = importances[indices]
            ordered_feature_names = self.X.columns[indices]
            # Save the results to a file and create the results directory if it doesn't exist
            os.makedirs(os.path.dirname(cache_file_path), exist_ok=True)
            with open(cache_file_path, "wb") as f:
                pickle.dump((ordered_importances, ordered_feature_names), f)

        # Generate bar plot
        sns.barplot(x=ordered_importances, y=ordered_feature_names)
        plt.title('Feature Importances')
        plt.savefig('plots/feature_importances.png', bbox_inches='tight')
        if self.show_plots:
            plt.show()

        # Generate a table of top features
        feature_importance_df = pd.DataFrame({
            'Feature': ordered_feature_names,
            'Importance': ordered_importances
        })
        if self.verbose:
            # Display top 10 features and their importances
            print(feature_importance_df.head(10))

    def univariate_selection(self, k: int = 10) -> pd.DataFrame:
        selector = SelectKBest(f_classif, k=k)
        X_new = selector.fit_transform(self.X, self.y)
        
        # Create a DataFrame for the F-values and sort the DataFrame by F-value in descending order
        f_values_df = pd.DataFrame({
            'Feature': self.X.columns,
            'F_value': selector.scores_
        })
        f_values_df = f_values_df.sort_values(by='F_value', ascending=False)
        
        if self.verbose:
            # Display top 10 features and their F-values
            print(f_values_df.head(10))
        
        return pd.DataFrame(X_new, columns=self.X.columns[selector.get_support()])
