# Import necessary libraries for data processing, modeling, and visualization
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# Load and merge cleaned COVID datasets
def load_merge_datasets():
    # Load cleaned CSVs
    covid_cases = pd.read_csv('cleaned_covid_cases.csv')
    vaccinations = pd.read_csv('cleaned_vaccinations.csv')
    testing_data = pd.read_csv('cleaned_testing_data.csv')

    # Merge all datasets on 'date' and 'state'
    merged_df = covid_cases.merge(vaccinations, on=['date', 'state'], how='left')
    merged_df = merged_df.merge(testing_data, on=['date', 'state'], how='left')

    # Ensure datetime format and proper sorting
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df = merged_df.sort_values(['state', 'date'])

    # Compute 7-day rolling average of new cases
    merged_df['new_cases'] = merged_df.groupby('state')['confirmed'].diff().rolling(7).mean()

    # Create lag features (previous 1-day and 7-day)
    merged_df['lag_1'] = merged_df.groupby('state')['new_cases'].shift(1)
    merged_df['lag_7'] = merged_df.groupby('state')['new_cases'].shift(7)

    # Fill missing values for sample and vaccination data
    merged_df['total_samples'] = merged_df['total_samples'].fillna(0)
    merged_df['first_dose'] = merged_df['first_dose'].fillna(0)

    # Compute positivity and vaccination rates, avoiding division by zero
    merged_df['positivity_rate'] = merged_df['new_cases'] / merged_df['total_samples'].replace(0, np.nan)
    merged_df['vaccination_rate'] = merged_df['first_dose'] / merged_df['total_samples'].replace(0, np.nan)

    # Create target variable: new cases 7 days ahead
    merged_df['target'] = merged_df.groupby('state')['new_cases'].shift(-7)

    # Drop rows with missing key inputs
    merged_df.dropna(subset=['lag_1', 'lag_7', 'positivity_rate', 'vaccination_rate', 'target'], inplace=True)

    return merged_df

# Load and prepare the data
data = load_merge_datasets()

# One-hot encode 'state' column for model input
data = pd.get_dummies(data, columns=['state'], drop_first=True)

# Define feature columns (lags, rates, and one-hot encoded states)
feature_cols = ['lag_1', 'lag_7', 'positivity_rate', 'vaccination_rate'] + \
               [col for col in data.columns if col.startswith('state_')]
X = data[feature_cols]
y = data['target']

# Split the dataset without shuffling (preserves temporal structure)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Scale features to [0,1] range using Min-Max Scaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost Regressor model
model = XGBRegressor(objective='reg:squarederror', n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Predict future cases and clip negatives to zero
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, None)

# Evaluate model using MAE and safe MAPE
mae = mean_absolute_error(y_test, predictions)
safe_denominator = np.maximum(np.abs(y_test), 10)  # Avoid division by very small numbers
mape = (np.abs(y_test - predictions) / safe_denominator).mean()
print(f'MAE: {mae:.2f}, Safe MAPE: {mape:.2%}')

# Plot actual vs predicted values
def visualize_predictions(actual, predicted, dates):
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=dates, y=actual, label='Actual', marker='o')
    sns.lineplot(x=dates, y=predicted, label='Predicted', marker='x')
    plt.xlabel('Date')
    plt.ylabel('New Cases')
    plt.title('Actual vs Predicted New Cases (7-Day Forecast)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Pass corresponding dates for the test set to the visualizer
visualize_predictions(y_test.values, predictions, data.loc[y_test.index, 'date'])
