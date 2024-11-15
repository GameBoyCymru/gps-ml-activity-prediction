import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
#from tabulate import tabulate

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, ".."))


# Function to load and label data
def load_activity_data(activity, folder_path):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tsv"):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
            df.columns = df.columns.str.strip().str.lower()
            if 'speed (km/h)' in df.columns:
                df['activity'] = activity
                data.append(df)
            else:
                print(f"Warning: 'Speed (km/h)' column missing in {file_name}")
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()

# Load data from each activity folder
activity_data = []
for activity, folder_path in [("Walking", "data/walking"),
                              ("Jogging", "data/jogging"),
                              ("Commuting", "data/commuting")]:
    full_path = os.path.join(parent_dir, folder_path)
    activity_data.append(load_activity_data(activity, full_path))

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

# Set up a smaller hyperparameter grid for GradientBoostingClassifier
param_grid = {
    'n_estimators': [50, 100],         # Number of boosting stages
    'learning_rate': [0.01, 0.1, 0.2],       # Learning rate shrinks the contribution of each tree
    'max_depth': [3, 5, 7],                  # Maximum depth of the individual estimators
    'min_samples_split': [2, 5, 10],         # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]            # Minimum number of samples required to be at a leaf node
}

# Use GridSearchCV with fewer cross-validation folds to speed up testing
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,  # Reduced to 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1
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

    return new_data, overall_activity

    #Display the first 20 row-by-row predictions in table format
    #print("\nFirst 20 row-by-row predictions:")
    #print(tabulate(result[['date', 'speed (km/h)', 'Predicted Activity']].head(15), headers='keys', tablefmt='pretty', showindex=False)


# Construct the path to the test_data.tsv file
test_data_path = os.path.join(parent_dir, "test_data.tsv")

result, overall_activity = predict_activity(test_data_path, best_model)


