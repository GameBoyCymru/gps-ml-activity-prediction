import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

 
activity_labels = {0: 'Standing Still', 1: 'Walking', 2: 'Jogging', 3: 'Commuting'}

# Prediction function for new data
def predict_activity(file_name, model):
    new_data = pd.read_csv(file_name, sep='\t', encoding='utf-16')
    new_data.columns = new_data.columns.str.strip().str.lower()
    if 'speed (km/h)' not in new_data.columns:
        raise KeyError("The new data file is missing the 'Speed (km/h)' column.")

    new_data = extract_features(new_data)
    X_new = new_data[['speed (km/h)', 'speed variance', 'avg speed', 'distance', 'acceleration', 'longitude', 'latitude']]

    # Normalize and reshape data
    X_new = scaler.transform(X_new)
    X_new = X_new.reshape(X_new.shape[0], X_new.shape[1], 1)

    predictions = model.predict(X_new)
    predicted_classes = encoder.inverse_transform(np.argmax(predictions, axis=1))

    new_data['Predicted Activity'] = predicted_classes
    overall_activity = new_data['Predicted Activity'].mode()[0]
    
    overall_activity = activity_labels[overall_activity]

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

# Decode labels
y_pred_decoded = [activity_labels[label] for label in y_pred] # Decode predicted labels to human-readable labels
y_true_decoded = [activity_labels[label] for label in y_test] # Decode true labels to human-readable labels

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

