import pandas as pd
import numpy as np
import os

class InsuranceDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target data file not found at: {file_path}")
            
    def load_and_initialize(self) -> pd.DataFrame:
        """Loads pipe-delimited data and safely injects key insurance KPI metrics."""
        # Force the engine to parse pipe characters and bypass irregular header anomalies
        df = pd.read_csv(
            self.file_path, 
            sep='|', 
            low_memory=False, 
            on_bad_lines='skip'
        )
        
        # Strip trailing white spaces from column titles
        df.columns = df.columns.str.strip()
        
        # Convert time variables safely
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            
        # Clean numeric data rows and calculate performance anchors
        if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
            df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce').fillna(0.0)
            df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0.0)
            
            # Loss Ratio calculation (handle division by zero via np.where)
            df['Loss_Ratio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], 0.0)
            # Margin calculation (Premium volume left over after claims)
            df['Margin'] = df['TotalPremium'] - df['TotalClaims']
            
        return df