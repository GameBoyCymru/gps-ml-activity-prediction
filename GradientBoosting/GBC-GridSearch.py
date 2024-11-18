import pandas as pd
import numpy as np
import os
from sklearn.ensemble import GradientBoostingClassifier
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
    new_data['Predicted Activity'] = new_data['Predicted Activity'].map(activity_labels)
    
    
    #Display the first 20 row-by-row predictions in table format
    #print("\nFirst 20 row-by-row predictions:")
    #print(tabulate(result[['date', 'speed (km/h)', 'Predicted Activity']].head(15), headers='keys', tablefmt='pretty', showindex=False)
    
    
    return overall_activity

    
    
# Construct the path to the test_data.tsv file
test_data_path = os.path.join(parent_dir, "test_data.tsv")
overall_activity = predict_activity(test_data_path, best_model)

print("\nOverall Predicted Activity for the file:", overall_activity)
print("\nBest Accuracy:", grid_search.best_score_)
print("\nBest Parameters:\n", grid_search.best_params_)
