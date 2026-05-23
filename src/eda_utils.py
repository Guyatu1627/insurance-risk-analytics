import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class InsuranceEDAVisualizer:
    @staticmethod
    def plot_numerical_distribution(df: pd.DataFrame, column: str, title: str = None):
        """Generates a clean distribution chart using a kernel density estimate."""
        plt.figure(figsize=(10, 4))
        sns.histplot(df[column], kde=True, color="#1a365d", bins=40)
        plt.title(title or f"Distribution Analysis for {column}")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_outlier_boxplot(df: pd.DataFrame, column: str, group_by: str = None):
        """Plots targeted boxplots to uncover distribution anomalies and outliers."""
        plt.figure(figsize=(10, 4))
        if group_by:
            sns.boxplot(data=df, x=group_by, y=column, palette="Set2")
            plt.xticks(rotation=45)
        else:
            sns.boxplot(data=df, y=column, color="#2b6cb0")
        plt.title(f"Outlier Boundary Inspection: {column}")
        plt.tight_layout()
        plt.show()