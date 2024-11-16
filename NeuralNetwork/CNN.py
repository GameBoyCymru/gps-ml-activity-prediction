import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Set options and paths
pd.set_option('future.no_silent_downcasting', True)
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

# Prepare data for CNN
X = df_combined[['speed (km/h)', 'speed variance', 'avg speed', 'distance']]
y = df_combined['activity']

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Reshape X for CNN (assuming 4 features per time step, reshape for time-series compatibility)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=35, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print("\nAccuracy:", accuracy)

# Prediction function for new data
def predict_activity(file_name, model):
    new_data = pd.read_csv(file_name, sep='\t', encoding='utf-16')
    new_data.columns = new_data.columns.str.strip().str.lower()
    if 'speed (km/h)' not in new_data.columns:
        raise KeyError("The new data file is missing the 'Speed (km/h)' column.")

    new_data = extract_features(new_data)
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance']]

    # Normalize and reshape data
    X_new = scaler.transform(X_new)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

    predictions = model.predict(X_new)
    predicted_classes = encoder.inverse_transform(np.argmax(predictions, axis=1))

    new_data['Predicted Activity'] = predicted_classes
    overall_activity = new_data['Predicted Activity'].mode()[0]

    print("\nOverall Predicted Activity for the file:", overall_activity)
    return new_data, overall_activity

# Construct the path to the test_data.tsv file
test_data_path = os.path.join(parent_dir, "test_data.tsv")

# Example usage with a new file
result, overall_activity = predict_activity(test_data_path, model)

# Get predictions for the test set
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class indices

# Decode true and predicted labels
y_true_decoded = encoder.inverse_transform(y_test)
y_pred_decoded = encoder.inverse_transform(y_pred)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true_decoded, y_pred_decoded, labels=encoder.classes_)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=encoder.classes_, yticklabels=encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
