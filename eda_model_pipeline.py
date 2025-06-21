import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports

# Visualization


class BusDemandForecaster:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}

    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        try:
            self.train_df = pd.read_csv('train/train.csv')
            self.test_df = pd.read_csv('test_8gqdJqH.csv')
            self.transactions_df = pd.read_csv('train/transactions.csv')

            # Convert date columns
            date_columns = ['doj', 'doi']
            for df in [self.train_df, self.test_df, self.transactions_df]:
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')

            # Remove rows with invalid dates
            self.train_df = self.train_df.dropna(subset=['doj'])
            self.test_df = self.test_df.dropna(subset=['doj'])
            self.transactions_df = self.transactions_df.dropna(subset=['doj'])

            print(f"Train shape: {self.train_df.shape}")
            print(f"Test shape: {self.test_df.shape}")
            print(f"Transactions shape: {self.transactions_df.shape}")

        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def create_holiday_features(self, df):
        """Create holiday and special date features"""
        df = df.copy()

        # Extract basic date features
        df['year'] = df['doj'].dt.year
        df['month'] = df['doj'].dt.month
        df['day'] = df['doj'].dt.day
        df['dayofweek'] = df['doj'].dt.dayofweek
        df['quarter'] = df['doj'].dt.quarter
        df['week_of_year'] = df['doj'].dt.isocalendar(
        ).week.astype(int)  # Convert to int
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        df['is_month_start'] = df['doj'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['doj'].dt.is_month_end.astype(int)

        # Festival seasons (approximate dates for major Indian festivals)
        df['is_diwali_season'] = (
            (df['month'] == 10) | (df['month'] == 11)).astype(int)
        df['is_holi_season'] = (df['month'] == 3).astype(int)
        df['is_dussehra_season'] = (df['month'] == 10).astype(int)
        df['is_wedding_season'] = (
            (df['month'] >= 11) | (df['month'] <= 2)).astype(int)
        df['is_summer_vacation'] = (
            (df['month'] >= 4) & (df['month'] <= 6)).astype(int)
        df['is_winter_vacation'] = (
            (df['month'] == 12) | (df['month'] == 1)).astype(int)

        # Long weekend indicators (approximate)
        df['is_potential_long_weekend'] = (
            (df['dayofweek'] == 4) | (df['dayofweek'] == 0)).astype(int)

        return df

    def create_lag_features(self, df, target_col=None, lag_days=[7, 14, 21, 30]):
        """Create lag features for time series"""
        df = df.copy()
        df = df.sort_values(['srcid', 'destid', 'doj'])

        if target_col and target_col in df.columns:
            for lag in lag_days:
                df[f'{target_col}_lag_{lag}'] = df.groupby(['srcid', 'destid'])[
                    target_col].shift(lag)

        return df

    def create_aggregated_features(self, df):
        """Create aggregated features from transactions data"""
        try:
            # Check if transactions data is available
            if self.transactions_df.empty:
                print("Warning: No transactions data available")
                return df

            # Aggregate at route level (15 days before journey)
            route_agg = self.transactions_df[self.transactions_df['dbd'] >= 15].groupby(['srcid', 'destid', 'doj']).agg({
                'cumsum_seatcount': ['max', 'mean', 'std'],
                'cumsum_searchcount': ['max', 'mean', 'std'],
                'dbd': ['min', 'max']
            }).reset_index()

            # Flatten column names
            route_agg.columns = ['srcid', 'destid', 'doj'] + [
                f'route_{col[0]}_{col[1]}' for col in route_agg.columns[3:]
            ]

            # Fill NaN values
            route_agg = route_agg.fillna(0)

            # Merge with main dataframe
            df = df.merge(route_agg, on=['srcid', 'destid', 'doj'], how='left')

            # Historical route performance
            historical_route = self.transactions_df.groupby(['srcid', 'destid']).agg({
                'cumsum_seatcount': ['mean', 'std', 'max'],
                'cumsum_searchcount': ['mean', 'std', 'max']
            }).reset_index()

            historical_route.columns = ['srcid', 'destid'] + [
                f'hist_route_{col[0]}_{col[1]}' for col in historical_route.columns[2:]
            ]

            df = df.merge(historical_route, on=['srcid', 'destid'], how='left')

        except Exception as e:
            print(f"Warning: Could not create aggregated features: {e}")

        return df

    def create_route_features(self, df):
        """Create route-specific features"""
        df = df.copy()

        # Route popularity (from training data)
        if hasattr(self, 'train_df'):
            route_popularity = self.train_df.groupby(['srcid', 'destid']).agg({
                'final_seatcount': ['mean', 'std', 'count']
            }).reset_index()

            route_popularity.columns = [
                'srcid', 'destid', 'route_avg_demand', 'route_std_demand', 'route_frequency']
            route_popularity['route_std_demand'] = route_popularity['route_std_demand'].fillna(
                0)

            df = df.merge(route_popularity, on=['srcid', 'destid'], how='left')

        # Encode categorical variables
        categorical_cols = ['srcid', 'destid']
        if 'srcid_region' in df.columns:
            categorical_cols.extend(
                ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier'])

        for col in categorical_cols:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        df[col].astype(str))
                else:
                    # Handle unseen categories
                    df[col] = df[col].astype(str)
                    mask = df[col].isin(self.encoders[col].classes_)
                    df[f'{col}_encoded'] = 0
                    df.loc[mask, f'{col}_encoded'] = self.encoders[col].transform(
                        df.loc[mask, col])

        return df

    def prepare_features(self, df, is_train=True):
        """Main feature engineering pipeline"""
        print("Creating features...")

        # Create holiday features
        df = self.create_holiday_features(df)

        # Create aggregated features from transactions
        df = self.create_aggregated_features(df)

        # Create route features
        df = self.create_route_features(df)

        # Create lag features for training data
        if is_train and 'final_seatcount' in df.columns:
            df = self.create_lag_features(df, 'final_seatcount')

        # Create interaction features
        df['month_dayofweek'] = df['month'] * 10 + df['dayofweek']
        df['is_peak_season'] = ((df['is_diwali_season'] == 1) |
                                (df['is_wedding_season'] == 1) |
                                (df['is_summer_vacation'] == 1)).astype(int)

        # Fill remaining NaN values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        return df

    def get_feature_columns(self, df):
        """Get list of feature columns for modeling"""
        exclude_cols = ['doj', 'route_key',
                        'final_seatcount', 'srcid', 'destid']
        if 'srcid_region' in df.columns:
            exclude_cols.extend(
                ['srcid_region', 'destid_region', 'srcid_tier', 'destid_tier'])

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models and select the best one"""
        print("Training models...")

        models_config = {
            'lightgbm': {
                'model': lgb.LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbosity=-1
                ),
                'params': {}
            },
            'xgboost': {
                'model': xgb.XGBRegressor(
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=6,
                    random_state=42,
                    verbosity=0
                ),
                'params': {}
            },
            'random_forest': {
                'model': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'params': {}
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'params': {}
            }
        }

        best_score = float('inf')
        best_model_name = None

        for name, config in models_config.items():
            print(f"Training {name}...")
            model = config['model']

            try:
                # Train model with updated API
                if name == 'lightgbm':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(
                            50), lgb.log_evaluation(0)]
                    )
                elif name == 'xgboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)

                # Validate
                y_pred = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))

                print(f"{name} RMSE: {rmse:.4f}")

                self.models[name] = model

                if rmse < best_score:
                    best_score = rmse
                    best_model_name = name

            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        if not self.models:
            # Fallback to a simple model if all fail
            print("All models failed, using simple Linear Regression")
            simple_model = LinearRegression()
            simple_model.fit(X_train, y_train)
            y_pred = simple_model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            self.models['linear'] = simple_model
            best_model_name = 'linear'
            best_score = rmse

        print(f"Best model: {best_model_name} with RMSE: {best_score:.4f}")
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name

        # Store feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = dict(
                zip(X_train.columns, self.best_model.feature_importances_))

    def create_ensemble_prediction(self, X):
        """Create ensemble prediction from multiple models"""
        predictions = []
        weights = {
            'lightgbm': 0.4,
            'xgboost': 0.3,
            'random_forest': 0.2,
            'gradient_boosting': 0.1
        }

        ensemble_pred = np.zeros(len(X))

        for name, model in self.models.items():
            pred = model.predict(X)
            ensemble_pred += weights.get(name, 0.1) * pred

        return ensemble_pred

    def train(self):
        """Main training pipeline"""
        # Load data
        self.load_data()

        # Prepare training data
        train_features = self.prepare_features(self.train_df, is_train=True)

        # Get feature columns
        feature_cols = self.get_feature_columns(train_features)

        # Check if we have any valid features
        if not feature_cols:
            print("Warning: No valid features found. Adding basic features.")
            train_features['basic_feature'] = 1
            feature_cols = ['basic_feature']

        # Prepare X and y
        X = train_features[feature_cols]
        y = train_features['final_seatcount']

        # Remove any infinite or NaN values
        mask = np.isfinite(X.select_dtypes(include=[np.number]).values).all(
            axis=1) & np.isfinite(y)
        X = X[mask]
        y = y[mask]

        print(f"Data after cleaning: {len(X)} rows")

        if len(X) == 0:
            raise ValueError("No valid data remaining after cleaning")

        # Time-based split for validation
        if 'doj' in train_features.columns:
            split_date = train_features['doj'].quantile(0.8)
            train_mask = train_features['doj'][mask] <= split_date
        else:
            # Fallback to simple split
            split_idx = int(0.8 * len(X))
            train_mask = np.zeros(len(X), dtype=bool)
            train_mask[:split_idx] = True

        X_train, X_val = X[train_mask], X[~train_mask]
        y_train, y_val = y[train_mask], y[~train_mask]

        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Number of features: {len(feature_cols)}")

        # Train models
        self.train_models(X_train, y_train, X_val, y_val)

        # Store feature columns for prediction
        self.feature_cols = feature_cols

    def predict(self):
        """Make predictions on test data"""
        print("Making predictions...")

        # Prepare test data
        test_features = self.prepare_features(self.test_df, is_train=False)

        # Ensure all required features are present
        missing_features = set(self.feature_cols) - set(test_features.columns)
        for feature in missing_features:
            test_features[feature] = 0

        X_test = test_features[self.feature_cols]

        # Make predictions using best model
        predictions = self.best_model.predict(X_test)

        # Ensure non-negative predictions
        predictions = np.maximum(predictions, 0)

        # Create submission file
        submission = pd.DataFrame({
            'route_key': self.test_df['route_key'],
            'final_seatcount': predictions
        })

        return submission

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        if self.feature_importance:
            importance_df = pd.DataFrame(
                list(self.feature_importance.items()),
                columns=['feature', 'importance']
            ).sort_values('importance', ascending=False).head(top_n)

            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(
                f'Top {top_n} Feature Importance ({self.best_model_name})')
            plt.tight_layout()
            plt.show()

    def run_full_pipeline(self):
        """Run the complete pipeline"""
        print("Starting Bus Demand Forecasting Pipeline...")
        print("=" * 50)

        # Train models
        self.train()

        # Make predictions
        submission = self.predict()

        # Save submission
        submission.to_csv('submission.csv', index=False)
        print("Submission saved to 'submission.csv'")

        # Plot feature importance
        try:
            self.plot_feature_importance()
        except:
            print("Could not plot feature importance")

        print("Pipeline completed successfully!")
        return submission

# Additional utility functions


def exploratory_data_analysis():
    """Perform basic EDA on the datasets"""
    print("Performing Exploratory Data Analysis...")

    # Load data
    train_df = pd.read_csv('train/train.csv')
    transactions_df = pd.read_csv('train/transactions.csv')

    # Convert dates
    train_df['doj'] = pd.to_datetime(train_df['doj'])
    transactions_df['doj'] = pd.to_datetime(transactions_df['doj'])
    transactions_df['doi'] = pd.to_datetime(transactions_df['doi'])

    print("Dataset Statistics:")
    print("-" * 30)
    print(f"Training data shape: {train_df.shape}")
    print(f"Transactions data shape: {transactions_df.shape}")
    print(
        f"Date range in training: {train_df['doj'].min()} to {train_df['doj'].max()}")
    print(f"Average seats booked: {train_df['final_seatcount'].mean():.2f}")
    print(
        f"Unique routes in training: {len(train_df[['srcid', 'destid']].drop_duplicates())}")

    # Plot demand distribution
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(train_df['final_seatcount'], bins=50, alpha=0.7)
    plt.title('Distribution of Final Seat Count')
    plt.xlabel('Seats Booked')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    monthly_demand = train_df.groupby(train_df['doj'].dt.month)[
        'final_seatcount'].mean()
    monthly_demand.plot(kind='bar')
    plt.title('Average Demand by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Seats Booked')

    plt.subplot(1, 3, 3)
    weekly_demand = train_df.groupby(train_df['doj'].dt.dayofweek)[
        'final_seatcount'].mean()
    weekly_demand.plot(kind='bar')
    plt.title('Average Demand by Day of Week')
    plt.xlabel('Day of Week (0=Monday)')
    plt.ylabel('Average Seats Booked')

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Optional: Run EDA first
    # exploratory_data_analysis()

    # Initialize and run the forecaster
    forecaster = BusDemandForecaster()
    submission = forecaster.run_full_pipeline()

    print("\nSubmission Preview:")
    print(submission.head())
    print(f"\nSubmission Statistics:")
    print(f"Total predictions: {len(submission)}")
    print(
        f"Average predicted demand: {submission['final_seatcount'].mean():.2f}")
    print(f"Min predicted demand: {submission['final_seatcount'].min():.2f}")
    print(f"Max predicted demand: {submission['final_seatcount'].max():.2f}")
