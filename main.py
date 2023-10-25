from data_loader import DataLoader
from prelim_analyser import PrelimAnalyser
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

    # Conduct preliminary analysis
    analyser = PrelimAnalyser(data=loader.data)
    analyser.generate_statistics()
    analyser.print_high_correlation_pairs()
    # analyser.plot_correlation_matrix()

if __name__ == "__main__":
    main()