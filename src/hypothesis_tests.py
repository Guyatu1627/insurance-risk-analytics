import pandas as pd
import numpy as np
from scipy import stats

class InsuranceHypothesisTester:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def run_chi_square_test(self, feature: str, target_kpi: str = 'has_claim') -> tuple:
        """Runs a Chi-Square Test of Independence for categorical features."""
        if target_kpi not in self.df.columns and target_kpi == 'has_claim':
            self.df['has_claim'] = self.df['TotalClaims'] > 0
            
        # Construct contingency mapping table
        contingency_table = pd.crosstab(self.df[feature], self.df[target_kpi])
        
        # Calculate chi2 and p-value metrics
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return chi2, p_value

    def run_two_sample_ttest(self, segment_col: str, group_a_val: str, group_b_val: str, target_kpi: str) -> tuple:
        """Runs an independent Welch's t-test for a numerical KPI between two groups."""
        group_a = self.df[self.df[segment_col] == group_a_val][target_kpi].dropna()
        group_b = self.df[self.df[segment_col] == group_b_val][target_kpi].dropna()
        
        # If testing claim severity, filter down strictly to positive claim events
        if target_kpi == 'TotalClaims':
            group_a = group_a[group_a > 0]
            group_b = group_b[group_b > 0]
            
        # Execute independent t-test assuming unequal population variances
        t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
        return t_stat, p_value