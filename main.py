from data_loader import DataLoader
from prelim_analyser import PrelimAnalyser
from feature_selector import FeatureSelector
import pandas as pd
import os

def main(verbose=True) -> None:
    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    loader = DataLoader(directory=data_dir, verbose=verbose)
    loader.cleanse_data()
    loader.add_binary_label(label_column='label', negative_class='BENIGN')
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(original_label_column='label', features=None, test_size=0.3, val_size=0.0)
    
    if verbose:
        pd.set_option('display.max_columns', 100)
        print(loader.data.head())
        print("Number of rows:", len(loader.data))
        print(loader.data.describe())

    # Conduct preliminary analysis
    analyser = PrelimAnalyser(data=X_train)
    analyser.generate_statistics()
    analyser.print_high_correlation_pairs()
    analyser.plot_correlation_matrix()

    # Feature selection
    selector = FeatureSelector(X_train, y_train)
    selector.model_based_selection(fresh_analysis=False)
    selector.univariate_selection()

    # count numeric and non-numeric columns
    numeric_cols = X_train.select_dtypes(include=['number']).columns
    non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Non-numeric columns: {len(non_numeric_cols)}")

if __name__ == "__main__":
    main()