# Activity Prediction based on GPS Data

## Overview

The project aims to predict activities based on GPS data using machine learning algorithms and neural networks. The dataset contains GPS data on four activities: `Commuting`, `Jogging`, `Walking` and `Idle`. The data is preprocessed to extract relevant features such as speed variance, average speed, distance, acceleration, jerk, and bearing change. The project includes the following models:

- Random Forest Classifier
- Support Vector Machine
- Gradient Boosting Classifier
- Convolutional Neural Network
- Recurrent Neural Network

## Project Structure

The project consists of the following main files:

    ┌── gps-ml-activity-prediction   
    │  ├── data
    │      ├── Commuting
    │      ├── Jogging
    │      ├── Walking
    │      └── load_data.py
    │  ├── GradientBoosting
    │      └── GradientBoostingClassifier.py
    │      └── GBC-CrossValidation.py
    │      └── GBC-GridSearch.py
    │  ├── NeuralNetwork
    │      └── CNN.py
    │      └── RNN.py
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

Each script contains a function to load and label data from multiple `.tsv` files (with `UTF-16` encoding). The data is then combined into a single DataFrame. Feature engineering is performed to extract relevant features such as speed variance, average speed, distance, acceleration, jerk, and bearing change.

- `load_data.py`: Contains functions to load and label data from multiple `.tsv` files (with `UTF-16` encoding). The data is then combined into a single DataFrame and returned to the calling script.

## Model Training and Evaluation

### Random Forest Classifier

- `RandomForestClassifier.py`: Trains a `Random Forest` model and evaluates its accuracy.
- `RFC-CrossValidation.py`: Performs 10-fold cross-validation on the `Random Forest` model.
- `RFC-GridSearch.py`: Uses GridSearch to find the best hyperparameters for the `Random Forest` model.

### Support Vector Machine

- `SupportVectorMachine.py`: Trains a `SVM` model and evaluates its accuracy.
- `SVM-CrossValidation.py`: Performs 10-fold cross-validation on the `SVM` model.
- `SVM-GridSearch.py`: Uses GridSearch to find the best hyperparameters for the `SVM` model.

### Gradient Boosting Classifier

- `GradientBoostingClassifier.py`: Trains a `Gradient Boosting` model and evaluates its accuracy.
- `GBC-CrossValidation.py`: Performs 10-fold cross-validation on the `Gradient Boosting` model.
- `GBC-GridSearch.py`: Uses GridSearch to find the best hyperparameters for the `Gradient Boosting` model.

### Neural Network

- `CNN.py`: Trains a `Convolutional Neural Network` and evaluates its accuracy, while also displaying a Confusion Matrix Heatmap.
- `RNN.py`: Trains a `Recurrent Neural Network` and evaluates its accuracy, while also displaying a Confusion Matrix Heatmap.

## Prediction

Each script includes a function to predict the activity for new data contained in a `.tsv` file (with `UTF-16` encoding). The function applies the same feature extraction process and uses the trained model to make predictions.

## Example Usage

To use any of the models for prediction, you can run any of the scripts (corresponding to the model), and modify the following line (may look slightly different) to provide the relative path to `.tsv` to test. The line will be located at the end of each script.

<sub>The `.tsv` must include the following columns:`Date`, `Longitude`, `Latitude`, `Speed (km/h)`.</sub>

```python
test_data_path = os.path.join(parent_dir, "test_data.tsv")
```

## Dependencies

Required **Python (3.12)** libraries:

- pandas
- numpy
- scikit-learn
- tensorflow (for Neural Networks)
- seaborn (for Neural Networks)
- matplotlib (for Neural Networks)

Optional Python libraries:

- tabulate (only required to print the results in a tabular format - disabled by default)

### Installing Dependencies via pip
```bash
pip install -r requirements.txt
```

## Conclusion

This project demonstrates the use of various machine learning algorithms and a neural network to predict activities based on GPS data. By utilizing techniques like GridSearch and CrossValidation, the models are optimized for better performance.
