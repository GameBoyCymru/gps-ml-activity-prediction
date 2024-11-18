import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
#from tabulate import tabulate

# Add the parent directory to the sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
from load_data import load_and_process_data
from load_data import extract_features 

pd.set_option('future.no_silent_downcasting', True)  # Hides downcasting warnings

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Load and process the data
X, y = load_and_process_data(parent_dir)

# Set up hyperparameter grid for SVC
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf'],
    'gamma': ['scale', 'auto']
}

# Use GridSearchCV to perform cross-validated hyperparameter tuning
grid_search = GridSearchCV(
    estimator=SVC(kernel='rbf', random_state=42, cache_size=500),
    param_grid=param_grid,
    cv=10,
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
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration', 'longitude', 'latitude']]

    # Make predictions for each row
    predictions = model.predict(X_new)
    new_data['Predicted Activity'] = predictions

    # Calculate the overall predicted activity for the file
    overall_activity = new_data['Predicted Activity'].mode()[0]  # Get the most frequent prediction
    
     
    activity_labels = {0: 'Standing Still', 1: 'Walking', 2: 'Jogging', 3: 'Commuting'}
    overall_activity = activity_labels[overall_activity]
    

    # Print overall activity
    print("\nOverall Predicted Activity for the file:", overall_activity)

    # Display the first 20 row-by-row predictions in table format
    #print("\nFirst 20 row-by-row predictions:")
    #print(tabulate(new_data[['date', 'speed (km/h)', 'Predicted Activity']].head(15), headers='keys', tablefmt='pretty', showindex=False))

    return new_data, overall_activity

# Construct the path to the test_data.tsv file
test_data_path = os.path.join(parent_dir, "test_data.tsv")

result, overall_activity = predict_activity(test_data_path, best_model)
