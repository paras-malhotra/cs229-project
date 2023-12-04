from data_loader import DataLoader
from prelim_analyser import PrelimAnalyser
from feature_selector import FeatureSelector
from binary_classifier import BinaryClassifier
from binary_classification_evaluator import BinaryClassificationEvaluator
import pandas as pd
import os

def main(verbose=True) -> None:
    # Load data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    loader = DataLoader(directory=data_dir, verbose=verbose)
    loader.load_and_clean_data(fresh=True)
    X_train, X_val, X_test, y_train, y_val, y_test, labels_train, labels_val, labels_test = loader.split_data(original_label_column='label', features=None, test_size=0.1, val_size=0.1)
    
    if verbose:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', 100)
        print(loader.data.head())
        print("Number of rows:", len(loader.data))
        print(loader.data.describe())
        print(loader.data['label'].value_counts())

    # Conduct preliminary analysis
    analyser = PrelimAnalyser(data=X_train)
    analyser.find_best_fit_distribution()
    analyser.generate_statistics()
    # analyser.test_normality()
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

    # Binary classification
    classifier = BinaryClassifier(X_train=X_train, y_train=y_train, verbose=verbose)
    models = classifier.train_models(retrain=True)

    # Evaluation
    evaluator = BinaryClassificationEvaluator(models=models, X_test=X_test, y_test=y_test, labels_test=labels_test, scaler=loader.scaler, verbose=verbose)
    evaluator.evaluate_models()
    evaluator.calculate_attack_identification_rate_by_label()
    evaluator.print_top_k_high_confidence_errors(k=100)

if __name__ == "__main__":
    main()