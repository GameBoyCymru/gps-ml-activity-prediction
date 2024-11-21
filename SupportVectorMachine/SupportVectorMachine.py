import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the classifier
model = SVC(kernel='rbf', random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)

# Prediction function for new data with formatted output
def predict_activity(file_name, model):
    

    new_data = pd.read_csv(file_name, sep='\t', encoding='utf-16')
    new_data.columns = new_data.columns.str.strip().str.lower()
    if 'speed (km/h)' not in new_data.columns:
        raise KeyError("The new data file is missing the 'Speed (km/h)' column.")

    # Apply feature extraction
    new_data = extract_features(new_data)
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration', 'longitude', 'latitude']]

    # Make predictions
    predictions = model.predict(X_new)
    new_data['Predicted Activity'] = predictions

    # Calculate the overall predicted activity
    overall_activity = new_data['Predicted Activity'].mode()[0]

     
    activity_labels = {0: 'Idle', 1: 'Walking', 2: 'Jogging', 3: 'Commuting'}
    overall_activity = activity_labels[overall_activity]
    new_data['Predicted Activity'] = new_data['Predicted Activity'].map(activity_labels)
    
    # Display the first 20 row-by-row predictions in table format
    #print("\nFirst 20 row-by-row predictions:")
    #print(tabulate(new_data[['date', 'speed (km/h)', 'Predicted Activity']].head(20), headers='keys', tablefmt='pretty', showindex=False))

    return overall_activity

# Construct the path to the test_data.tsv file
test_data_path = os.path.join(parent_dir, "test_data.tsv")

overall_activity = predict_activity(test_data_path, model)

print("\nOverall Predicted Activity for the file:", overall_activity)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
