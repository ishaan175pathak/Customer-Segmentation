import numpy
from pandas import DataFrame, Series, read_csv
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
from script.feature_builder import UserFeatureBuilder
from script.visualizer import DataVisualizer
from script.splitter import DataSplitter
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from pathlib import Path
import os


os.system('cls')

# importing the dataset

CSV_FILE: str = "mobile_app_interactions_expanded.csv"

# Building useful features from the dataset
dataset: DataFrame = UserFeatureBuilder(fileName=CSV_FILE).__build_user_level_dataset__()

# printing the head
print(dataset.head())

# initializing the visualizer
dataVisualizer: DataVisualizer = DataVisualizer(dataset, save_dir="visual_refs")

dataVisualizer.plot_class_distribution()
dataVisualizer.plot_missing_values()
dataVisualizer.plot_numeric_histograms()
dataVisualizer.plot_correlation_heatmap()

# splitting the dataset

X_train: DataFrame
X_test: DataFrame
y_train: Series
y_test: Series

(X_train, X_test, y_train, y_test) = DataSplitter(dataset).__split_dataset__()

# visualizing the dataset

dataVisualizer.plot_pca_2d(X_train, y_train)

# check if the shape of training and validation sets are same or not

assert X_train.shape[1] == X_test.shape[1], "Train and Validation dataset shape does not match with each other"

# Using a KMeans Algorithm with n = 5 (default), 11, 21

k_values = [5, 11, 21]
results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    results.append({
        "k": k,
        "model": knn,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "y_test_pred": y_test_pred
    })

for result in results:
    print(
        f"K={result['k']} | "
        f"Train Accuracy={result['train_accuracy']:.4f} | "
        f"Test Accuracy={result['test_accuracy']:.4f}"
    )

# evaluating the results

for result in results:
    y_pred = result["y_test_pred"]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    misclassification_error = 1 - accuracy

    print(f"\nK = {result['k']}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Misclassification Error: {misclassification_error:.4f}")

# visualizing the results after training the models
dataVisualizer.plot_model_evaluation(results, y_test)