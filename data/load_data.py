import os
import pandas as pd
import numpy as np

# Define the activities and their corresponding folder paths
ACTIVITIES = {
    "Walking": "data/walking",
    "Jogging": "data/jogging",
    "Commuting": "data/commuting",
}

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

    # Check if 'activity' column exists
    if 'activity' not in df_combined.columns:
        raise KeyError("'activity' column is missing from the combined dataset.")

    # Feature engineering
    df_combined = df_combined.groupby('activity', group_keys=False).apply(
        lambda x: extract_features(x).assign(activity=x.name)
    )

    # Prepare X and y
    X = df_combined[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration']]
    y = df_combined['activity']
    return X, y
