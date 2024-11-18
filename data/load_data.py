import os
import pandas as pd
import numpy as np

# Define the activities and their corresponding folder paths
ACTIVITIES = {
    "1": "data/walking",    # "1" Corresponds to the labelled data for walking
    "2": "data/jogging",    # "2" Corresponds to the labelled data for jogging
    "3": "data/commuting",  # "3" Corresponds to the labelled data for commuting
}

def load_activity_data(activity, folder_path):
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv") or file_name.endswith(".tsv"):
            file_path = os.path.join(folder_path, file_name)
            if file_name.endswith(".csv"):
                df = pd.read_csv(file_path, sep=',', encoding='utf-8')
            else:
                df = pd.read_csv(file_path, sep='\t', encoding='utf-16')
            df.columns = df.columns.str.strip().str.lower()
            # if no target column, add it
            if 'target' not in df.columns:
                df['target'] = activity
            data.append(df)
    return pd.concat(data, ignore_index=True) if data else pd.DataFrame()


def extract_features(df):
    df['speed variance'] = df['speed (km/h)'].rolling(window=5, min_periods=1).std()
    df['avg speed'] = df['speed (km/h)'].rolling(window=5, min_periods=1).mean()
    df['distance'] = np.sqrt((df['longitude'].diff() ** 2 + df['latitude'].diff() ** 2))
    df['acceleration'] = df['speed (km/h)'].diff() 
    df = df.fillna(0)
    df = df.infer_objects()
    return df


def load_and_process_data(base_path):
    activity_data = []
    # Loop through predefined activities (defined above in ACTIVITIES)
    for activity, folder_path in ACTIVITIES.items():
        full_path = os.path.join(base_path, folder_path)
        activity_data.append(load_activity_data(activity, full_path))

    # Combine all labelled data into one DataFrame
    df_combined = pd.concat(activity_data, ignore_index=True)
    
     # Load labeled data
    labeled_data = load_activity_data(0, os.path.join(os.path.dirname(__file__), os.path.join(os.path.dirname(__file__), "Labelled")))

    # Combine data
    df_combined = pd.concat([df_combined, labeled_data], ignore_index=True)


    # Check if 'activity' column exists
    # if 'activity' not in df_combined.columns:
    #     raise KeyError("'activity' column is missing from the combined dataset.")

    # Feature engineering
    df_combined = df_combined.groupby('target', group_keys=False).apply(
        lambda x: extract_features(x).assign(target=int(x.name)))

    # Prepare X and y
    X = df_combined[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration', 'longitude', 'latitude']]
    y = df_combined['target']
    return X, y
