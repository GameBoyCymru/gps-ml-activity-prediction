import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical

pd.set_option('future.no_silent_downcasting', True)  # Hides downcasting warnings

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

# Combine all labelled data into one DataFrame
df_combined = pd.concat(activity_data, ignore_index=True)

if 'activity' not in df_combined.columns:
    raise KeyError("'activity' column is missing from df_combined.")

# Feature engineering function
def extract_features(df):
    df['speed variance'] = df['speed (km/h)'].rolling(window=5).std()
    df['avg speed'] = df['speed (km/h)'].rolling(window=5).mean()
    df['distance'] = np.sqrt((df['longitude'].diff() ** 2 + df['latitude'].diff() ** 2))
    df = df.fillna(0)
    return df

df_combined = df_combined.groupby('activity', group_keys=False).apply(
    lambda x: extract_features(x).assign(activity=x.name))

# Prepare data for model training
X = df_combined[['speed (km/h)', 'speed variance', 'avg speed', 'distance']].values
y = df_combined['activity'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Reshape input for RNN (samples, timesteps, features)
timesteps = 10
num_samples = len(X_normalized) // timesteps
X_reshaped = X_normalized[:num_samples * timesteps].reshape(num_samples, timesteps, -1)
y_reshaped = y_categorical[:num_samples * timesteps:timesteps]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Define the RNN model
model = Sequential([
    LSTM(64, input_shape=(timesteps, X_reshaped.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(32, return_sequences=False),
    Dropout(0.3),
    Dense(y_reshaped.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=35, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nAccuracy:", accuracy)

# Prediction function for new data
def predict_activity(file_name, model, label_encoder, scaler, timesteps=10):
    new_data = pd.read_csv(file_name, sep='\t', encoding='utf-16')
    new_data.columns = new_data.columns.str.strip().str.lower()

    if 'speed (km/h)' not in new_data.columns:
        raise KeyError("The new data file is missing the 'Speed (km/h)' column.")

    # Apply feature extraction
    new_data = extract_features(new_data)
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance']].values

    # Normalize and reshape data
    X_new_normalized = scaler.transform(X_new)
    num_samples = len(X_new_normalized) // timesteps
    X_new_reshaped = X_new_normalized[:num_samples * timesteps].reshape(num_samples, timesteps, -1)

    # Predict activities
    predictions = model.predict(X_new_reshaped)
    predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))

    # Determine overall activity
    overall_activity = pd.Series(predicted_labels).mode()[0]
    print("\nOverall Predicted Activity for the file:", overall_activity)

    return predicted_labels, overall_activity

# Example usage
test_data_path = os.path.join(parent_dir, "test_data.tsv")
predicted_labels, overall_activity = predict_activity(test_data_path, model, label_encoder, scaler)
