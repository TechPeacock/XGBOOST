# xgboost_model.py
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

class SwingTradingXGBoost:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Select and prepare features for XGBoost"""
        feature_cols = [
            # Price features
            'price_change_pct', 'high_low_spread_pct', 'close_to_high',
            'price_position', 'open_close_spread',
            
            # Volume features
            'DELIV_PER', 'delivery_strength', 'avg_trade_size',
            'NO_OF_TRADES', 'TURNOVER_LACS',
            
            # Technical features
            'Annual_Vol', 'Log_Returns',
            
            # 52-week features
            'distance_from_52w_high', 'distance_from_52w_low', '52w_range_position',
            
            # Risk features
            'risk_score',
            
            # Multi-day features
            'delivery_trend', 'volume_trend'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        # Handle missing values
        df[available_features] = df[available_features].fillna(0)
        
        # Replace infinities
        df[available_features] = df[available_features].replace([np.inf, -np.inf], 0)
        
        return df[available_features]
    
    def create_labels(self, df, future_days=5, threshold=0.03):
        """Create swing trading labels based on future price movement"""
        df['future_return'] = df.groupby('SYMBOL')['CLOSE_PRICE'].shift(-future_days) / df['CLOSE_PRICE'] - 1
        
        # Create categories
        conditions = [
            df['future_return'] >= threshold,  # BUY signal
            df['future_return'] <= -threshold,  # SELL signal
        ]
        choices = ['BUY', 'SELL']
        df['signal'] = np.select(conditions, choices, default='HOLD')
        
        return df
    
    def train_model(self, X_train, y_train):
        """Train XGBoost model"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # XGBoost parameters optimized for swing trading
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'random_state': 42
        }
        
        self.model = xgb.XGBClassifier(**params)
        
        # Train with early stopping
        eval_set = [(X_train, y_encoded)]
        self.model.fit(
            X_train, y_encoded,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        return self.model
    
    def predict(self, X):
        """Make predictions with probability scores"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Get probability predictions
        proba = self.model.predict_proba(X)
        
        # Get class predictions
        predictions = self.model.predict(X)
        labels = self.label_encoder.inverse_transform(predictions)
        
        # Create DataFrame with probabilities
        prob_df = pd.DataFrame(proba, columns=['prob_BUY', 'prob_HOLD', 'prob_SELL'])
        prob_df['prediction'] = labels
        prob_df['confidence'] = proba.max(axis=1)
        
        return prob_df
    
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_imp
    
    def save_model(self, path):
        """Save trained model"""
        if self.model:
            joblib.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns
            }, path)
    
    def load_model(self, path):
        """Load saved model"""
        saved = joblib.load(path)
        self.model = saved['model']
        self.label_encoder = saved['label_encoder']
        self.feature_columns = saved['feature_columns']