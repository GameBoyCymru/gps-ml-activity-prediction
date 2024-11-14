import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from tabulate import tabulate

# Function to load and label data from multiple files in each folder
def load_activity_data(activity, folder_path):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tsv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, sep='\t', encoding='utf-16')  # Load TSV with proper encoding
            df.columns = df.columns.str.strip().str.lower()  # Standardize column names
            if 'speed (km/h)' in df.columns:
                df['activity'] = activity  # Label the activity
                data.append(df)
            else:
                print(f"Warning: 'Speed (km/h)' column missing in {file_name}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

# Load and label data from each activity folder
activity_data = []
for activity, folder_path in [("Walking", "walking"),
                              ("Jogging", "jogging"),
                              ("Commuting", "commuting")]:
    activity_data.append(load_activity_data(activity, folder_path))

# Combine all labeled data into one DataFrame
df_combined = pd.concat(activity_data, ignore_index=True)

# Check if 'activity' column exists and print column names for debugging
if 'activity' not in df_combined.columns:
    raise KeyError("'activity' column is missing from df_combined.")
#print("Columns in df_combined:", df_combined.columns)

# Enhanced Feature Engineering Function
def extract_features(df):
    df['speed variance'] = df['speed (km/h)'].rolling(window=5, min_periods=1).std()
    df['avg speed'] = df['speed (km/h)'].rolling(window=5, min_periods=1).mean()
    df['distance'] = np.sqrt((df['longitude'].diff() ** 2 + df['latitude'].diff() ** 2))
    df['acceleration'] = df['speed (km/h)'].diff()  # New feature: speed change between readings
    df = df.fillna(0)
    df = df.infer_objects()
    return df

df_combined = df_combined.groupby('activity', group_keys=False).apply(
    lambda x: extract_features(x).assign(activity=x.name))

# Prepare data for model training
X = df_combined[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration']]
y = df_combined['activity']

# Set up hyperparameter grid for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200, 300],       # Number of trees in the forest
    'max_depth': [10, 20, None],           # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],       # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]          # Minimum number of samples required to be at a leaf node
}

# Use GridSearchCV to perform cross-validated hyperparameter tuning
grid_search = GridSearchCV(
    estimator=SVC(kernel='rbf', random_state=42),
    param_grid=param_grid,
    cv=10,  # 10-fold cross-validation
    scoring='accuracy',
    n_jobs=-1  # Use all available CPU cores
)

# Fit GridSearchCV to the data to find the best model
grid_search.fit(X, y)

# Print the best parameters and the corresponding score
print("\nBest Parameters:\n", grid_search.best_params_)
print("\nBest Cross-Validation Accuracy:", grid_search.best_score_)

# Use the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Prediction function for new data with formatted output
def predict_activity(file_name, model):
    new_data = pd.read_csv(file_name, sep='\t', encoding='utf-16')
    new_data.columns = new_data.columns.str.strip().str.lower()  # Standardize column names
    if 'speed (km/h)' not in new_data.columns:
        raise KeyError("The new data file is missing the 'Speed (km/h)' column.")

    # Apply feature extraction
    new_data = extract_features(new_data)
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration']]

    # Make predictions for each row
    predictions = model.predict(X_new)
    new_data['Predicted Activity'] = predictions

    # Calculate the overall predicted activity for the file
    overall_activity = new_data['Predicted Activity'].mode()[0]  # Get the most frequent prediction

    # Print overall activity
    print("\nOverall Predicted Activity for the file:", overall_activity)

    # Display the first 20 row-by-row predictions in table format
    #print("\nFirst 20 row-by-row predictions:")
    #print(tabulate(new_data[['date', 'speed (km/h)', 'Predicted Activity']].head(15), headers='keys', tablefmt='pretty', showindex=False))

    return new_data, overall_activity

# Example usage with a new file
# Since GridSearchCV has already fit the best model, we use it directly for prediction
result, overall_activity = predict_activity("test_data.tsv", best_model)
