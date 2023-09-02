import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from utils import dataloader, modelloader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    multilabel_confusion_matrix,
)
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

# This is a test!
def grid_search_random_forest():
    n_estimators = [25]
    max_depth = [25]
    min_samples_leaf = [2]
    bootstrap = [True, False]

    param_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "bootstrap": bootstrap,
    }


def grid_search_knn():
    knn = KNeighborsClassifier()

    k_range = list(range(1, 31))
    param_grid = dict(n_neighbors=k_range)

    # defining parameter range
    grid = GridSearchCV(
        knn, param_grid, cv=10, scoring="accuracy", return_train_score=False, verbose=1
    )

    # fitting the model for grid search
    grid_search = grid.fit(x_train, y_train)

    print(grid_search.best_params_)

    accuracy = grid_search.best_score_ * 100
    print("Accuracy for our training dataset with tuning is : {:.2f}%".format(accuracy))


def best_k_value():
    # How to decide the right k-value for the dataset
    neighbors = np.arange(1, 30)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over K values
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)

        # Compute training and test data accuracy
        train_accuracy[i] = knn.score(x_train, y_train)
        test_accuracy[i] = knn.score(x_test, y_test)

    # Generate plot
    plt.plot(neighbors, test_accuracy, label="Testing dataset Accuracy")
    plt.plot(neighbors, train_accuracy, label="Training dataset Accuracy")

    plt.legend()
    plt.xlabel("n_neighbors")
    plt.ylabel("Accuracy")
    plt.show()
    plt.savefig("figures/knn_plot.png", dpi=200)


def gradient_boosting_training():
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


def gnb_training():
    model = GaussianNB()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


def random_forest_training():
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


def tree_training():
    model = tree.DecisionTreeClassifier()
    model = model.fit(x_train, y_train)

    # Calculate the accuracy of the model
    predictions = model.predict(x_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


def knn_training():
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    # Calculate the accuracy of the model
    predictions = model.predict(x_test)
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    x, y = dataloader.load_dataset(
        dataset="Drone_Dataset/0823_10_classes", feature="mfcc"
    )
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42
    )

    print(y)
    print(len(np.unique(y)))
    print(x_train.shape)

    knn_training()
