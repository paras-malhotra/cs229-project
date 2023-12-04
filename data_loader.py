import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List
import glob
import os
import numpy as np

class DataLoader:
    def __init__(self, directory: str, verbose: bool = True):
        self.verbose = verbose
        self.directory = directory

    def load_and_clean_data(self, fresh: bool = False) -> None:
        cleaned_data_path = os.path.join(self.directory, 'cleaned.csv')

        if not fresh and os.path.exists(cleaned_data_path):
            # Load cleaned data from file if it exists and fresh is False
            self.data = pd.read_csv(cleaned_data_path)
            if self.verbose:
                print(f'Loaded cleaned data from {cleaned_data_path}')
        else:
            if os.path.exists(cleaned_data_path):
                os.remove(cleaned_data_path)
                if self.verbose:
                    print(f'Deleted existing cleaned data at {cleaned_data_path}')

            # Find all CSV files in the specified directory
            file_paths = glob.glob(os.path.join(self.directory, '*.csv'))

            data_list = []
            for file_path in file_paths:
                df = pd.read_csv(file_path)
                data_list.append(df)

            # Load and concatenate data from all found CSV files with source info
            self.data = pd.concat(data_list, ignore_index=True)
            self.data.columns = self.data.columns.str.strip().str.lower().str.replace(' ', '_')

            initial_size = len(self.data)
            self.data.replace([np.inf, -np.inf], np.nan, inplace=True)
            # Removing rows with any missing values
            self.data.dropna(inplace=True)
            # Filter numeric columns
            numeric_data = self.data.select_dtypes(include=['number'])
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

            self.add_binary_label(label_column='label', negative_class='BENIGN')
            self.data.to_csv(cleaned_data_path, index=False)
            if self.verbose:
                print(f'Saved cleaned data to {cleaned_data_path}')

    def add_binary_label(self, label_column: str, negative_class: str) -> None:
        self.data['binary_label'] = self.data[label_column].apply(lambda x: 0 if x == negative_class else 1)

    def split_data(self, original_label_column: str, features: List[str] = None, test_size: float = 0.3, val_size: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        # If features is None, use all columns except original_label_column and 'binary_label'
        if features is None:
            features = [col for col in self.data.columns if col not in [original_label_column, 'binary_label']]

        X = self.data[features]
        y = self.data['binary_label']
        # Identify numeric columns
        numeric_cols = X.select_dtypes(include=['number']).columns

        if val_size > 0:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size + val_size, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size/(test_size + val_size), random_state=42)

            # Identify numeric columns
            numeric_cols = X_train.select_dtypes(include='number').columns

            # Scale the numeric columns
            self.scaler = StandardScaler()
            X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
            X_val[numeric_cols] = self.scaler.transform(X_val[numeric_cols])
            X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Identify numeric columns
            numeric_cols = X_train.select_dtypes(include='number').columns

            # Scale the numeric columns
            self.scaler = StandardScaler()
            X_train[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            X_val = None
            y_val = None

        labels_train = self.data.loc[X_train.index, original_label_column]
        labels_val = None if val_size == 0 else self.data.loc[X_val.index, original_label_column]
        labels_test = self.data.loc[X_test.index, original_label_column]
        # self.scaledData = pd.concat([X_train, X_val, X_test], ignore_index=True)

        return X_train, X_val, X_test, y_train, y_val, y_test, labels_train, labels_val, labels_test