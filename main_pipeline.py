# main_pipeline_enhanced.py - Enhanced version with XGBoost and date selection
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib


class NSEDataProcessor:
    def __init__(self, base_path="./data"):
        self.base_path = Path(base_path)
        self.date_folders = self.find_date_folders()

        if self.date_folders:
            print(f"Found {len(self.date_folders)} date folders:")
            for folder in self.date_folders[-5:]:  # Show last 5
                print(f"  - {folder.name}")

    def find_date_folders(self):
        """Find all date folders"""
        date_folders = []
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir() and len(item.name) == 8 and item.name.isdigit():
                    date_folders.append(item)
        return sorted(date_folders)

    def load_data_for_date(self, date_str):
        """Load data for a specific date"""
        date_path = self.base_path / date_str / "Equity"

        if not date_path.exists():
            print(f"No data found for date: {date_str}")
            return pd.DataFrame()

        # Load bhavdata
        bhav_file = date_path / f"sec_bhavdata_full_{date_str}.csv"

        if bhav_file.exists():
            print(f"Loading: {bhav_file.name}")
            df = pd.read_csv(bhav_file)

            # Clean columns
            df.columns = df.columns.str.strip()

            # Convert numeric columns
            numeric_cols = ['PREV_CLOSE', 'OPEN_PRICE', 'HIGH_PRICE', 'LOW_PRICE',
                            'CLOSE_PRICE', 'TTL_TRD_QNTY', 'TURNOVER_LACS',
                            'NO_OF_TRADES', 'DELIV_PER']

            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add date column
            df['DATE'] = date_str

            return df

        return pd.DataFrame()

    def load_multiple_dates(self, num_days=5):
        """Load data for multiple recent dates"""
        all_data = []

        # Get last n date folders
        recent_folders = self.date_folders[-num_days:] if len(self.date_folders) >= num_days else self.date_folders

        for folder in recent_folders:
            df = self.load_data_for_date(folder.name)
            if not df.empty:
                all_data.append(df)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"Loaded {len(all_data)} days of data, total {len(combined_df)} rows")
            return combined_df

        return pd.DataFrame()


class FeatureEngineer:
    def create_features(self, df):
        """Create comprehensive features for XGBoost"""
        if df.empty:
            return df

        # Price features
        if 'CLOSE_PRICE' in df.columns and 'PREV_CLOSE' in df.columns:
            df['price_change'] = df['CLOSE_PRICE'] - df['PREV_CLOSE']
            df['price_change_pct'] = (df['price_change'] / (df['PREV_CLOSE'] + 0.001)) * 100
            df['price_momentum'] = df['price_change_pct'].rolling(3, min_periods=1).mean()

        # Volume features
        if 'TTL_TRD_QNTY' in df.columns:
            df['volume_change'] = df.groupby('SYMBOL')['TTL_TRD_QNTY'].pct_change()
            df['volume_ma_ratio'] = df['TTL_TRD_QNTY'] / df.groupby('SYMBOL')['TTL_TRD_QNTY'].transform(
                lambda x: x.rolling(5, min_periods=1).mean())

        # Delivery features
        if 'DELIV_PER' in df.columns:
            df['delivery_strength'] = df['DELIV_PER'] / 100
            df['high_delivery'] = (df['DELIV_PER'] > 60).astype(int)
            df['delivery_trend'] = df.groupby('SYMBOL')['DELIV_PER'].transform(
                lambda x: x.rolling(3, min_periods=1).mean())

        # Price range features
        if all(col in df.columns for col in ['HIGH_PRICE', 'LOW_PRICE', 'CLOSE_PRICE']):
            df['intraday_range'] = (df['HIGH_PRICE'] - df['LOW_PRICE']) / (df['LOW_PRICE'] + 0.001) * 100
            df['close_position'] = (df['CLOSE_PRICE'] - df['LOW_PRICE']) / (df['HIGH_PRICE'] - df['LOW_PRICE'] + 0.001)
            df['upper_shadow'] = (df['HIGH_PRICE'] - df['CLOSE_PRICE']) / (df['HIGH_PRICE'] - df['LOW_PRICE'] + 0.001)
            df['lower_shadow'] = (df['CLOSE_PRICE'] - df['LOW_PRICE']) / (df['HIGH_PRICE'] - df['LOW_PRICE'] + 0.001)

        # Trade characteristics
        if 'NO_OF_TRADES' in df.columns and 'TTL_TRD_QNTY' in df.columns:
            df['avg_trade_size'] = df['TTL_TRD_QNTY'] / (df['NO_OF_TRADES'] + 1)
            df['trade_intensity'] = df['NO_OF_TRADES'] / df['NO_OF_TRADES'].mean()

        # Turnover features
        if 'TURNOVER_LACS' in df.columns:
            df['turnover_per_trade'] = df['TURNOVER_LACS'] / (df['NO_OF_TRADES'] + 1)
            df['price_volume_ratio'] = df['CLOSE_PRICE'] * df['TTL_TRD_QNTY'] / (df['TURNOVER_LACS'] * 100000 + 1)

        return df


class XGBoostSwingModel:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None

    def prepare_data(self, df):
        """Prepare data for XGBoost"""
        # Select feature columns
        feature_cols = [
            'price_change_pct', 'price_momentum', 'volume_change', 'volume_ma_ratio',
            'delivery_strength', 'high_delivery', 'delivery_trend',
            'intraday_range', 'close_position', 'upper_shadow', 'lower_shadow',
            'avg_trade_size', 'trade_intensity', 'turnover_per_trade',
            'DELIV_PER', 'NO_OF_TRADES', 'TURNOVER_LACS'
        ]

        # Keep only available columns
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features

        # Fill missing values
        X = df[available_features].fillna(0)

        # Replace infinities
        X = X.replace([np.inf, -np.inf], 0)

        return X

    def create_labels(self, df, future_periods=5, threshold=0.03):
        """Create labels based on future price movement"""
        if 'CLOSE_PRICE' not in df.columns:
            return None

        # Calculate future returns
        df['future_return'] = df.groupby('SYMBOL')['CLOSE_PRICE'].transform(
            lambda x: x.shift(-future_periods) / x - 1
        )

        # Create labels
        conditions = [
            df['future_return'] > threshold,  # BUY
            df['future_return'] < -threshold,  # SELL
        ]
        choices = ['BUY', 'SELL']

        df['label'] = np.select(conditions, choices, default='HOLD')

        return df['label']

    def train(self, X, y):
        """Train XGBoost model"""
        # Remove rows with NaN labels
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) == 0:
            print("No valid training data")
            return

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        # XGBoost parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.01,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'random_state': 42
        }

        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.2%}")

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Important Features:")
        for idx, row in importance.head(10).iterrows():
            print(f"  {row['feature']:20} : {row['importance']:.4f}")

        return self.model

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            return None, None

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Decode labels
        labels = self.label_encoder.inverse_transform(predictions)

        # Get confidence (max probability)
        confidence = probabilities.max(axis=1)

        return labels, confidence

    def save_model(self, filepath="xgboost_swing_model.pkl"):
        """Save trained model"""
        if self.model:
            joblib.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_columns': self.feature_columns
            }, filepath)
            print(f"Model saved to: {filepath}")

    def load_model(self, filepath="xgboost_swing_model.pkl"):
        """Load saved model"""
        try:
            saved = joblib.load(filepath)
            self.model = saved['model']
            self.label_encoder = saved['label_encoder']
            self.feature_columns = saved['feature_columns']
            print(f"Model loaded from: {filepath}")
            return True
        except:
            return False


class SwingTradingSystem:
    def __init__(self):
        self.processor = NSEDataProcessor()
        self.engineer = FeatureEngineer()
        self.xgb_model = XGBoostSwingModel()

    def train_model(self, num_days=10):
        """Train XGBoost model on historical data"""
        print("\n" + "=" * 60)
        print("TRAINING XGBOOST MODEL")
        print("=" * 60)

        # Load multiple days of data
        df = self.processor.load_multiple_dates(num_days)

        if df.empty:
            print("No data available for training")
            return

        # Create features
        df = self.engineer.create_features(df)

        # Prepare data
        X = self.xgb_model.prepare_data(df)
        y = self.xgb_model.create_labels(df)

        # Train model
        self.xgb_model.train(X, y)

        # Save model
        self.xgb_model.save_model()

    def predict_latest(self):
        """Make predictions on latest data"""
        print("\n" + "=" * 60)
        print("GENERATING PREDICTIONS")
        print("=" * 60)

        # Load latest date
        if not self.processor.date_folders:
            print("No data folders found")
            return pd.DataFrame()

        latest_date = self.processor.date_folders[-1].name
        df = self.processor.load_data_for_date(latest_date)

        if df.empty:
            print(f"No data for {latest_date}")
            return pd.DataFrame()

        print(f"Loaded {len(df)} stocks from {latest_date}")

        # Create features
        df = self.engineer.create_features(df)

        # Try to load existing model
        if not self.xgb_model.model:
            if not self.xgb_model.load_model():
                print("No saved model found. Training new model...")
                self.train_model()

        # Prepare features
        X = self.xgb_model.prepare_data(df)

        # Make predictions
        predictions, confidence = self.xgb_model.predict(X)

        if predictions is not None:
            df['prediction'] = predictions
            df['confidence'] = confidence

            # Sort by confidence
            df = df.sort_values('confidence', ascending=False)

            return df

        return pd.DataFrame()

    def display_results(self, df):
        """Display trading signals"""
        if df.empty or 'prediction' not in df.columns:
            print("No predictions available")
            return

        print("\n" + "=" * 60)
        print("TRADING SIGNALS")
        print("=" * 60)

        # Signal distribution
        signal_counts = df['prediction'].value_counts()
        print("\n📊 Signal Distribution:")
        for signal, count in signal_counts.items():
            print(f"  {signal}: {count} stocks ({count / len(df) * 100:.1f}%)")

        # Top BUY signals
        print("\n🟢 Top 15 BUY Signals:")
        buy_signals = df[df['prediction'] == 'BUY'].head(15)

        for idx, row in buy_signals.iterrows():
            print(f"  {row['SYMBOL']:15} | ₹{row['CLOSE_PRICE']:8.2f} | "
                  f"Del: {row['DELIV_PER']:5.1f}% | "
                  f"Conf: {row['confidence']:.1%}")

        # Top SELL signals
        print("\n🔴 Top 10 SELL Signals:")
        sell_signals = df[df['prediction'] == 'SELL'].head(10)

        for idx, row in sell_signals.iterrows():
            print(f"  {row['SYMBOL']:15} | ₹{row['CLOSE_PRICE']:8.2f} | "
                  f"Del: {row['DELIV_PER']:5.1f}% | "
                  f"Conf: {row['confidence']:.1%}")

        # Save results
        output_file = f"xgboost_signals_{datetime.now().strftime('%Y%m%d')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to: {output_file}")


if __name__ == "__main__":
    print("NSE Swing Trading System - XGBoost Edition")
    print(f"User: TechPeacock | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize system
    system = SwingTradingSystem()

    # Menu
    print("\n" + "=" * 60)
    print("SELECT OPTION:")
    print("=" * 60)
    print("1. Train new XGBoost model")
    print("2. Generate predictions using saved model")
    print("3. Train and predict")

    choice = input("\nEnter choice (1-3): ").strip()

    if choice == '1':
        days = input("Enter number of days for training (default 10): ").strip()
        days = int(days) if days.isdigit() else 10
        system.train_model(days)

    elif choice == '2':
        df = system.predict_latest()
        system.display_results(df)

    elif choice == '3':
        system.train_model(10)
        df = system.predict_latest()
        system.display_results(df)

    else:
        # Default: just predict
        df = system.predict_latest()
        system.display_results(df)