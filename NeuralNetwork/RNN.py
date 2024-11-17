import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt

# Add the parent directory to the sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
from load_data import load_and_process_data
from load_data import extract_features 

pd.set_option('future.no_silent_downcasting', True)  # Hides downcasting warnings

parent_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Load and process the data
X, y = load_and_process_data(parent_dir)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Reshape input for RNN
timesteps = 10
num_samples = len(X_normalized) // timesteps
X_reshaped = X_normalized[:num_samples * timesteps].reshape(num_samples, timesteps, -1)
y_reshaped = y_categorical[:num_samples * timesteps:timesteps]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_reshaped, test_size=0.2, random_state=42)

# Defines the RNN model
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
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration']].values

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


# Predict activity for test data
test_data_path = os.path.join(parent_dir, "test_data.tsv")
predicted_labels, overall_activity = predict_activity(test_data_path, model, label_encoder, scaler)

# Get predictions
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)  # Get class indices for predictions
y_true = y_test.argmax(axis=1)       # Get class indices for true labels

# Decode labels
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_true_decoded = label_encoder.inverse_transform(y_true)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_true_decoded, y_pred_decoded, labels=label_encoder.classes_)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Activity')
plt.ylabel('True Activity')
plt.title('RNN Confusion Matrix Heatmap')
plt.show()


