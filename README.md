# Activitiy Prediction based on GPS Data

## Overview

This project aims to predict the type of activity (e.g., Walking, Jogging, Commuting) based on GPS data contained in `.tsv` files. The project utilizes three machine learning algorithms and a neural network to perform the predictions. The machine learning algorithms used are:

1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. Gradient Boosting Classifier

Additionally, GridSearch and CrossValidation techniques are applied to each of these algorithms to optimize their performance. A Long Short-Term Memory (LSTM) neural network is also implemented for activity prediction. For all algorithms, the accuracy is printed to the console, alongside the activity the algorithm has predicted for the file. Optionally, Tabulate can be used to also display the prediction on a row by row basis all Machine Learning models.

## Project Structure

The project consists of the following main files:

    ┌── gps-ml-activity-prediction   
    │  ├── data
    │      ├── Commuting
    │      ├── Jogging
    │      └── Walking
    │  ├── GradientBoosting
    │      └── GradientBoostingClassifier.py
    │      └── GBC-CrossValidation.py
    │      └── GBC-GridSearch.py
    │  ├── NeuralNetwork
    │      └── LSTM.py
    │  ├── RandomForest
    │      └── RandomForestClassifier.py
    │      └── RFC-CrossValidation.py
    │      └── RFC-GridSearch.py
    │  ├── SupportVectorMachine
    │      └── SupportVectorMachine.py
    │      └── SVM-CrossValidation.py
    │      └── SVM-GridSearch.py
    │  └── README.md
    │  └── test_data.tsv
    └────────────────────────────────

## Data Loading and Preprocessing

Each script contains a function to load and label data from multiple `.tsv` files. The data is then combined into a single DataFrame. Feature engineering is performed to extract relevant features such as speed variance, average speed, distance, acceleration, jerk, and bearing change.

## Model Training and Evaluation

### Random Forest Classifier

- `RandomForestClassifier.py`: Trains a `Random Forest` model and evaluates its accuracy.
- `RFC-CrossValidation.py`: Performs 10-fold cross-validation on the `Random Forest` model.
- `RFC-GridSearch.py`: Uses GridSearch to find the best hyperparameters for the `Random Forest` model.

### Support Vector Machine

- `SupportVectorMachine.py`: Trains an `SVM` model and evaluates its accuracy.
- `SVM-CrossValidation.py`: Performs 10-fold cross-validation on the `SVM` model.
- `SVM-GridSearch.py`: Uses GridSearch to find the best hyperparameters for the `SVM` model.

### Gradient Boosting Classifier

- `GradientBoostingClassifier.py`: Trains a `Gradient Boosting` model and evaluates its accuracy.
- `GBC-CrossValidation.py`: Performs 10-fold cross-validation on the `Gradient Boosting` model.
- `GBC-GridSearch.py`: Uses GridSearch to find the best hyperparameters for the `Gradient Boosting` model.

### LSTM Neural Network

- `LSTM.py`: Trains an LSTM neural network and evaluates its accuracy.

## Prediction

Each script includes a function to predict the activity for new data contained in a `.tsv` file. The function applies the same feature extraction process and uses the trained model to make predictions.

## Example Usage

To use any of the models for prediction, you can run any of the scripts (corresponding to the model), and modify the following line (may look slightly different) to provide the path to `.tsv` to test. The line will be located at the end of each script.

<sub>The `.tsv` must include the following columns:`Date`, `Longitude`, `Latitude`, `Speed (km/h)`.</sub>

```python
result, overall_activity = predict_activity("test_data.tsv", model)
```

## Dependencies

Required Python libraries:
- pandas
- numpy
- scikit-learn
- tensorflow (for LSTM)

Optional Python libraries:
- tabulate (only required to print the results in a tabular format - disabled by default)

You can install these dependencies using pip:

```bash
pip install pandas scikit-learn tensorflow tabulate
```

## Conclusion

This project demonstrates the use of various machine learning algorithms and a neural network to predict activities based on GPS data. By utilizing techniques like GridSearch and CrossValidation, the models are optimized for better performance.
