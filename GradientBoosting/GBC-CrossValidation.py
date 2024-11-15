import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
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

# Feature engineering function
def extract_features(df):
    df['speed variance'] = df['speed (km/h)'].rolling(window=5, min_periods=1).std()
    df['avg speed'] = df['speed (km/h)'].rolling(window=5, min_periods=1).mean()
    df['distance'] = np.sqrt((df['longitude'].diff() ** 2 + df['latitude'].diff() ** 2))
    df = df.fillna(0)
    df = df.infer_objects()
    return df

df_combined = df_combined.groupby('activity', group_keys=False).apply(
    lambda x: extract_features(x).assign(activity=x.name))

# Prepare data for model training
X = df_combined[['speed (km/h)', 'speed variance', 'avg speed', 'distance']]
y = df_combined['activity']

# Create the GBC model
model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Perform 10-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=10, scoring='accuracy')

# Print out the cross-validation scores and the mean accuracy
print("\nCross-Validation Scores:\n", cv_scores)
print("\nAverage Cross-Validation Accuracy:", np.mean(cv_scores))

# Prediction function for new data with formatted output
def predict_activity(file_name, model):
    new_data = pd.read_csv(file_name, sep='\t', encoding='utf-16')
    new_data.columns = new_data.columns.str.strip().str.lower()  # Standardize column names
    if 'speed (km/h)' not in new_data.columns:
        raise KeyError("The new data file is missing the 'Speed (km/h)' column.")

    # Apply feature extraction
    new_data = extract_features(new_data)
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance']]

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

model.fit(X, y)

# Construct the path to the test_data.tsv file
test_data_path = os.path.join(parent_dir, "test_data.tsv")

result, overall_activity = predict_activity(test_data_path, model)
