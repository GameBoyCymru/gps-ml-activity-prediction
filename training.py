import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the data
def load_data(activity, folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"The directory '{folder}' does not exist.")
    # display the names of all csv files in the folder
    print(f"Files in '{folder}': {os.listdir(folder)}")

    # the first row contains the column names
    data = pd.read_csv(f'{folder}/{os.listdir(folder)[0]}')
    data['Activity'] = activity
    return data



# Load the data
walking = load_data('Walking', 'walking')
running = load_data('jogging', 'jogging')
standing = load_data('commuting', 'commuting')