import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class InsurancePredictiveModeler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.model = None
        self.X_cols = None
        
    def prepare_features_and_targets(self, target_col: str = 'TotalClaims') -> tuple:
        """Cleans and encodes data efficiently by removing high-cardinality blockers."""
        # Baseline structural age tracking
        if 'Vehicle_Age' not in self.df.columns and 'RegistrationYear' in self.df.columns:
            self.df['Vehicle_Age'] = 2026 - self.df['RegistrationYear']
            self.df['Vehicle_Age'] = self.df['Vehicle_Age'].clip(lower=0)

        # CRITICAL: Drop extreme high-cardinality features causing one-hot explosion
        # Also drop secondary leakage indicators
        columns_to_drop = [
            'UnderwrittenCoverID', 'PolicyID', 'TransactionMonth', 
            'VehicleIntroDate', 'PostalCode', 'MainDrivingLicenceMMYY'
        ]
        existing_drops = [c for c in columns_to_drop if c in self.df.columns]
        working_df = self.df.drop(columns=existing_drops)
        
        # Additionally drop other unique ID markers or text fields if they exist
        high_card_text = [col for col in working_df.select_dtypes(include=['object']).columns 
                          if working_df[col].nunique() > 50]
        working_df = working_df.drop(columns=high_card_text)
        
        # Define targets
        X = working_df.drop(columns=[target_col, 'Loss_Ratio', 'Margin'], errors='ignore')
        y = working_df[target_col]
        
        # One-hot encode the safe, low-cardinality features
        X_encoded = pd.get_dummies(X, drop_first=True)
        self.X_cols = X_encoded.columns
        
        return X_encoded, y

    def train_random_forest(self, X, y) -> tuple:
        """Trains an optimized, highly-accelerated Random Forest configuration."""
        # Split data cleanly
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # OPTIMIZATION: Dropped n_estimators to 15 and max_depth to 6 for immediate execution
        # n_jobs=-1 forces utilization of every available CPU core in your machine
        self.model = RandomForestRegressor(
            n_estimators=15, 
            max_depth=6, 
            max_features='sqrt',
            random_state=42, 
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate performance
        y_pred = self.model.predict(X_test)
        
        metrics = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2_Score": r2_score(y_test, y_pred)
        }
        
        feature_imp_df = pd.DataFrame({'Feature': self.X_cols, 'Importance': self.model.feature_importances_})
        feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)
        
        return metrics, feature_imp_df