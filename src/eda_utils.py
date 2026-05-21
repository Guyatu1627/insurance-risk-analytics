import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class InsuranceEDAVisualizer:
    @staticmethod
    def plot_univariate_distributions(df: pd.DataFrame, column: str, is_categorical: bool = False):
        """Generates distribution plots based on feature layout style."""
        plt.figure(figsize=(10, 5))
        if is_categorical:
            order = df[column].value_counts().index[:15] # Top 15 categories for readability
            sns.countplot(data=df, x=column, order=order, palette="viridis")
            plt.xticks(rotation=45, ha='right')
            plt.title(f"Categorical Value Distribution: {column}")
        else:
            sns.histplot(df[column], kde=True, color="#1a365d", bins=40)
            plt.title(f"Numerical Frequency Distribution: {column}")
            
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_outlier_boxplots(df: pd.DataFrame, column: str, group_by: str = None):
        """Isolates outlier ranges using boxplots."""
        plt.figure(figsize=(10, 5))
        if group_by:
            sns.boxplot(data=df, x=group_by, y=column, palette="Set2")
            plt.xticks(rotation=45)
        else:
            sns.boxplot(data=df, y=column, color="#2b6cb0")
        plt.title(f"Outlier Range Analysis: {column}")
        plt.tight_layout()
        plt.show()