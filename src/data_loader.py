import pandas as pd
import numpy as np
import os

class InsuranceDataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Target insurance data file missing at: {file_path}")
            
    def load_and_initialize(self) -> pd.DataFrame:
        """Loads pipe-delimited auto insurance data and creates performance metrics."""
        # 1. Force the engine to parse pipe delimiters ('|')
        # 2. Use on_bad_lines='skip' to cleanly handle the compressed row 1 header anomaly
        df = pd.read_csv(
            self.file_path, 
            sep='|', 
            low_memory=False, 
            on_bad_lines='skip'
        )
        
        # Clean column spaces if any exist from the source export
        df.columns = df.columns.str.strip()
        
        # Cast target datetime objects safely [cite: 26]
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'], errors='coerce')
            
        # Compute Anchored Performance Metrics [cite: 27]
        if 'TotalClaims' in df.columns and 'TotalPremium' in df.columns:
            # Enforce numeric conversion in case of text artifacts
            df['TotalPremium'] = pd.to_numeric(df['TotalPremium'], errors='coerce').fillna(0.0)
            df['TotalClaims'] = pd.to_numeric(df['TotalClaims'], errors='coerce').fillna(0.0)
            
            # Loss Ratio = TotalClaims / TotalPremium [cite: 28]
            df['Loss_Ratio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], 0.0)
            
            # Margin = TotalPremium - TotalClaims [cite: 29]
            df['Margin'] = df['TotalPremium'] - df['TotalClaims']
            
        return df