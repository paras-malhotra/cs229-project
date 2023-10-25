import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import glob
import os
import numpy as np

class DataLoader:
    def __init__(self, directory: str, verbose: bool = True):
        self.verbose = verbose
        # Find all CSV files in the specified directory
        file_paths = glob.glob(os.path.join(directory, '*.csv'))

        data_list = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            # df['source_file'] = os.path.basename(file_path)
            data_list.append(df)

        # Load and concatenate data from all found CSV files with source info
        self.data = pd.concat(data_list, ignore_index=True)
        # Strip whitespace, convert to lowercase, and replace spaces with underscores in column names
        self.data.columns = self.data.columns.str.strip().str.lower().str.replace(' ', '_')

    def cleanse_data(self) -> None:
        initial_size = len(self.data)
        # Removing rows with any missing values
        self.data.dropna(inplace=True)

        # Filter numeric columns
        numeric_data = self.data.select_dtypes(include=[np.number])
        # Capture columns with constant values (zero variance)
        constant_columns = numeric_data.columns[numeric_data.std() == 0]
        # Drop these columns from the original dataset
        self.data.drop(columns=constant_columns, inplace=True)

        final_size = len(self.data)
        rows_removed = initial_size - final_size
        if self.verbose:
            print(f"Removed {rows_removed} rows with missing values.")
            # Print the columns dropped
            if constant_columns.size > 0:
                print("Dropped columns with constant values:", ", ".join(constant_columns))

    def add_binary_label(self, label_column: str, negative_class: str) -> None:
        self.data['binary_label'] = self.data[label_column].apply(lambda x: 0 if x == negative_class else 1)

    def split_data(self, original_label_column: str, features: List[str] = None, test_size: float = 0.3, val_size: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        # If features is None, use all columns except original_label_column and 'binary_label'
        if features is None:
            features = [col for col in self.data.columns if col not in [original_label_column, 'binary_label']]

        X = self.data[features]
        y = self.data['binary_label']

        if val_size > 0:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(test_size + val_size), random_state=42)
            return X_train, X_val, X_test, y_train, y_val, y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            # Returning None for X_val and y_val when val_size is 0
            return X_train, None, X_test, y_train, None, y_test