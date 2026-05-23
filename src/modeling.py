import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

try:
    import shap
except ModuleNotFoundError:
    shap = None

class InsurancePredictiveModeler:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        # 10 Academy Requirement: Cross-compare Linear models, Tree Ensembles, and Boosting
        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=15, max_depth=6, random_state=42, n_jobs=-1),
            "XGBoost": xgb.XGBRegressor(n_estimators=15, max_depth=5, learning_rate=0.1, random_state=42)
        }
        
    def prepare_features_and_targets(self, target_col: str = 'TotalClaims') -> tuple:
        """Removes unique text indicators and encodes metadata features cleanly."""
        columns_to_drop = ['UnderwrittenCoverID', 'PolicyID', 'TransactionMonth', 'VehicleIntroDate', 'PostalCode']
        working_df = self.df.drop(columns=[c for c in columns_to_drop if c in self.df.columns])
        
        # Eliminate high-cardinality columns with more than 50 options to prevent matrix bloating
        high_card = [c for c in working_df.select_dtypes(include=['object']).columns if working_df[c].nunique() > 50]
        working_df = working_df.drop(columns=high_card)
        
        X = working_df.drop(columns=[target_col, 'Loss_Ratio', 'Margin'], errors='ignore')
        y = pd.to_numeric(working_df[target_col], errors='coerce')
        
        X_encoded = pd.get_dummies(X, drop_first=True)
        X_encoded = X_encoded.dropna(axis=1, how='all')

        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_encoded),
            columns=X_encoded.columns,
            index=X_encoded.index,
        )
        
        non_missing = y.notna()
        return X_imputed.loc[non_missing], y.loc[non_missing]

    def evaluate_all_models(self, X, y) -> pd.DataFrame:
        """Trains all 3 challenge targets and outputs benchmarking scores."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        results = []
        
        for name, model in self.models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            results.append({"Model": name, "RMSE": rmse, "R2_Score": r2})
            
        return pd.DataFrame(results)

    def generate_shap_explainers(self, X):
        """Builds a tree explainer matrix to analyze feature importance weights."""
        if shap is None:
            raise ImportError(
                "SHAP is required for SHAP explainer generation. "
                "Install the shap package and restart the notebook kernel."
            )
        X_train, _, y_train, _ = train_test_split(X, self.df['TotalClaims'], test_size=0.3, random_state=42)
        best_model = self.models["XGBoost"].fit(X_train, y_train)
        
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X)
        return explainer, shap_values