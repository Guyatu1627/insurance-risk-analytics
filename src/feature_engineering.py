import pandas as pd
import numpy as np

class InsuranceFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def handle_data_quality(self) -> 'InsuranceFeatureEngineer':
        """Detects and mitigates missing data using structural column defaults."""
        # 1. For numerical columns, fill missing inputs with their median values
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna(self.df[col].median())

        # 2. For categorical columns, handle missing metrics with an explicit 'Unknown' flag
        cat_cols = self.df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col] = self.df[col].fillna('Unknown')
                
        return self

    def engineer_underwriting_signals(self) -> pd.DataFrame:
        """Derives actionable underwriting risk indicators from raw features."""
        # Feature 1: Historical Vehicle Age at the Exact Transaction Moment
        if 'RegistrationYear' in self.df.columns and 'TransactionMonth' in self.df.columns:
            self.df['Vehicle_Age_At_Transaction'] = self.df['TransactionMonth'].dt.year - self.df['RegistrationYear']
            # Clamp potential negative or corrupt data entries to 0
            self.df['Vehicle_Age_At_Transaction'] = self.df['Vehicle_Age_At_Transaction'].clip(lower=0)

        # Feature 2: Anti-Theft Security Density Score (0 to 2 Scale)
        if 'AlarmImmobiliser' in self.df.columns and 'TrackingDevice' in self.df.columns:
            has_alarm = (self.df['AlarmImmobiliser'] == 'Yes').astype(int)
            has_tracker = (self.df['TrackingDevice'] == 'Yes').astype(int)
            self.df['Security_Device_Count'] = has_alarm + has_tracker

        # Feature 3: Insured Asset Premium-To-Value Concentration Intensity
        if 'SumInsured' in self.df.columns and 'TotalPremium' in self.df.columns:
            self.df['Premium_To_Value_Ratio'] = np.where(
                self.df['SumInsured'] > 0, 
                self.df['TotalPremium'] / self.df['SumInsured'], 
                0.0
            )

        return self.df