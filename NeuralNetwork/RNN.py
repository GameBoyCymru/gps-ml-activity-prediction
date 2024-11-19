import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Hide TensorFlow warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
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
history = model.fit(X_train, y_train, epochs=35, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)


activity_labels = {0: 'Standing Still', 1: 'Walking', 2: 'Jogging', 3: 'Commuting'}

# Prediction function for new data
def predict_activity(file_name, model, label_encoder, scaler, timesteps=10):
    new_data = pd.read_csv(file_name, sep='\t', encoding='utf-16')
    new_data.columns = new_data.columns.str.strip().str.lower()

    if 'speed (km/h)' not in new_data.columns:
        raise KeyError("The new data file is missing the 'Speed (km/h)' column.")

    # Apply feature extraction
    new_data = extract_features(new_data)
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration', 'longitude', 'latitude']].values

    # Normalize and reshape data
    X_new_normalized = scaler.transform(X_new)
    num_samples = len(X_new_normalized) // timesteps
    X_new_reshaped = X_new_normalized[:num_samples * timesteps].reshape(num_samples, timesteps, -1)

    # Predict activities
    predictions = model.predict(X_new_reshaped)
    predicted_labels = label_encoder.inverse_transform(predictions.argmax(axis=1))

    # Determine overall activity
    overall_activity = pd.Series(predicted_labels).mode()[0]
    
    # Print the activity code to a human-readable label (without changing the original data)
    
    overall_activity = activity_labels[overall_activity] 
    
    return overall_activity

# Predict activity for test data
test_data_path = os.path.join(parent_dir, "test_data.tsv")
overall_activity = predict_activity(test_data_path, model, label_encoder, scaler)

# Get predictions
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)  # Get class indices for predictions
y_true = y_test.argmax(axis=1)       # Get class indices for true labels

# Decode labels
y_pred_decoded = [activity_labels[label] for label in y_pred] # Decode predicted labels to human-readable labels
y_true_decoded = [activity_labels[label] for label in y_true] # Decode true labels to human-readable labels


print("\nOverall Predicted Activity for the file:", overall_activity)
print("\nAccuracy:", accuracy)
print("\nLoss:", loss)
# Generate and display classification report
print("\nClassification Report:\n")
print(classification_report(y_true_decoded, y_pred_decoded, target_names=list(activity_labels.values())))


# Generate confusion matrix
conf_matrix = confusion_matrix(y_true_decoded, y_pred_decoded, labels=list(activity_labels.values()))

# Plot confusion matrix with labels
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=list(activity_labels.values()), 
            yticklabels=list(activity_labels.values()))
plt.xlabel('Predicted Activity')
plt.ylabel('True Activity')
plt.title('Confusion Matrix with Activity Labels')
plt.show()

# Plot Loss and Accuracy Curves
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.show()


