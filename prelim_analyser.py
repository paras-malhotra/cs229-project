import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from fitter import Fitter
import os

class PrelimAnalyser:
    def __init__(self, data: pd.DataFrame, show_plots: bool = False):
        self.data = data
        self.show_plots = show_plots

    def generate_statistics(self) -> None:
        print(self.data.describe())

    def plot_distributions(self) -> None:
        for column in self.get_numerical_data().columns:
            sns.histplot(self.data[column], kde=True)
            plt.title(f'Distribution of {column}')
            plt.show()
            plt.clf()

    def find_best_fit_distribution(self, fresh_analysis: bool = False) -> pd.DataFrame:
        results_path = os.path.join('results', 'distribution_table.txt')

        if not fresh_analysis and os.path.exists(results_path):
            distribution_table = pd.read_csv(results_path, sep='\t')
            print(distribution_table)
            return distribution_table

        distribution_names = []
        feature_names = []
        for column in self.get_numerical_data().columns:
            # Sample 10k data points to speed up the process
            data = self.data[column].sample(n=10000, random_state=42)
            f = Fitter(data, timeout=60, distributions='common')
            f.fit()
            best_fit_dict = f.get_best()
            best_fit_name, best_fit_params = max(best_fit_dict.items(), key=lambda x: x[1])
            print(f'Best fit for {column}: {best_fit_name}')
            feature_names.append(column)
            distribution_names.append(best_fit_name)

        distribution_table = pd.DataFrame({
            'Feature': feature_names,
            'Distribution': distribution_names
        })
        distribution_table.to_csv(results_path, sep='\t', index=False)
        print(distribution_table)

        return distribution_table

    def test_normality(self) -> None:
        for column in self.get_numerical_data().columns:
            stat, p_value = shapiro(self.data[column])
            print(f'Shapiro-Wilk Test for {column}: Statistic={stat:.3f}, p-value={p_value:.3f}')

    def print_high_correlation_pairs(self, threshold: float = 0.4) -> None:
        correlation_matrix = self.get_numerical_data().corr()
        # Extract the upper triangle of the matrix without the diagonal
        upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        upper_tri = upper_tri.dropna()
        # Find index pairs with correlation above the threshold and sort them
        high_corr_pairs = sorted(
            [(column, row) for column in upper_tri.columns for row in upper_tri.index if abs(upper_tri[column][row]) > threshold],
            key=lambda x: abs(upper_tri[x[0]][x[1]]), reverse=True
        )
        # Print the pairs and their correlation
        if not high_corr_pairs:
            print("No pairs found with a correlation above the specified threshold.")
        else:
            for pair in high_corr_pairs:
                print(f"Correlation between {pair[0]} and {pair[1]}: {upper_tri[pair[1]][pair[0]]:.2f}")


    def plot_correlation_matrix(self) -> None:
        # Select only the numerical columns of the DataFrame
        correlation_matrix = self.get_numerical_data().corr()
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.savefig('plots/correlation_matrix.png', bbox_inches='tight')
        if self.show_plots:
            plt.show()
        plt.clf()
    
    def get_numerical_data(self) -> pd.DataFrame:
        # Select only the numerical columns of the DataFrame
        numerical_data = self.data.select_dtypes(include=['number'])
        # Filter out columns with no variance
        numerical_data = numerical_data.loc[:, numerical_data.std() > 0]
        valid_columns = numerical_data.columns[numerical_data.nunique() != 1]
        numerical_data = numerical_data[valid_columns]
        return numerical_data
