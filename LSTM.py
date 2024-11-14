import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

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
for activity, folder_path in [("Walking", "walking"),
                              ("Jogging", "jogging"),
                              ("Commuting", "commuting")]:
    activity_data.append(load_activity_data(activity, folder_path))

# Combine all labeled data into one DataFrame
df_combined = pd.concat(activity_data, ignore_index=True)

# Feature extraction function
def extract_features(df):
    df['speed variance'] = df['speed (km/h)'].rolling(window=5, min_periods=1).std()
    df['avg speed'] = df['speed (km/h)'].rolling(window=5, min_periods=1).mean()
    df['distance'] = np.sqrt((df['longitude'].diff() ** 2 + df['latitude'].diff() ** 2))
    df['acceleration'] = df['speed (km/h)'].diff()
    df['jerk'] = df['acceleration'].diff()
    df['bearing change'] = np.arctan2(df['latitude'].diff(), df['longitude'].diff()).diff()
    df = df.fillna(0)
    return df

# Apply feature extraction
df_combined = df_combined.groupby('activity', group_keys=False).apply(
    lambda x: extract_features(x).assign(activity=x.name)
)

# Encode activity labels
le = LabelEncoder()
df_combined['activity'] = le.fit_transform(df_combined['activity'])

# Scaling features for deep learning
scaler = MinMaxScaler()
feature_cols = ['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration', 'jerk', 'bearing change']
df_combined[feature_cols] = scaler.fit_transform(df_combined[feature_cols])

# Split the data into sequences (with labels for training and just features for prediction)
def create_sequences(df, window_size=50, is_predict=False):
    sequences = []
    labels = []
    for i in range(0, len(df) - window_size, window_size):
        seq = df[feature_cols].iloc[i:i + window_size].values
        sequences.append(seq)
        if not is_predict:  # Only add labels when not predicting
            label = df['activity'].iloc[i + window_size - 1]
            labels.append(label)
    sequences = np.array(sequences)
    if not is_predict:
        labels = np.array(labels)
        return sequences, labels
    else:
        return sequences


# Prepare data for model
X, y = create_sequences(df_combined)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=55, batch_size=19, validation_split=0.2)

# Evaluate the model
y_pred = np.argmax(model.predict(X_test), axis=1)
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# Prediction function for new data
def predict_activity(file_name, model, le, scaler, window_size=50):
    new_data = pd.read_csv(file_name, sep='\t', encoding='utf-16')
    new_data.columns = new_data.columns.str.strip().str.lower()
    if 'speed (km/h)' not in new_data.columns:
        raise KeyError("The new data file is missing the 'Speed (km/h)' column.")

    # Apply feature extraction
    new_data = extract_features(new_data)
    new_data[feature_cols] = scaler.transform(new_data[feature_cols])  # Scale features

    # Create sequences for prediction without needing the 'activity' column
    sequences = create_sequences(new_data, window_size=window_size, is_predict=True)

    # Make predictions for each sequence
    predictions = model.predict(sequences)
    predicted_labels = [le.inverse_transform([np.argmax(pred)])[0] for pred in predictions]

    # Get the most frequent predicted activity
    overall_activity = max(set(predicted_labels), key=predicted_labels.count)
    print("\nOverall Predicted Activity for the file:", overall_activity)

    return predicted_labels, overall_activity

predicted_labels, overall_activity = predict_activity("test_data.tsv", model, le, scaler)
