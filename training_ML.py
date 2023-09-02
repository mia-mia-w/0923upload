import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow import keras
from utils import dataloader, modelloader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    multilabel_confusion_matrix,
)
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# This is a test!

if __name__ == "__main__":
    # load data and apply preprocessing method
    x, y = dataloader.load_dataset(
        dataset="Drone_Dataset/0308_22_classes", feature="mfcc"
    )
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    print(x_train.shape[0])
    print(x_train.shape[1])
    print(x_train.shape)

    # GradientBoostingClassifier
    GBC = GradientBoostingClassifier().fit(x_train, y_train)
    print(GBC.score(x_test, y_test))

    #GaussianNB
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    print(gnb.score(x_test, y_test))


    #Decision Tree
    tree = tree.DecisionTreeClassifier()
    tree = tree.fit(x_train, y_train)
    print(tree.score(x_test, y_test))


    #Random Forest
    forest = RandomForestClassifier()
    forest = forest.fit(x_train, y_train)
    print(forest.score(x_test, y_test))

   
    # knn classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn = knn.fit(x_train, y_train)
    # print(knn.predict(x_test))
    print(knn.score(x_test, y_test))

    # print(y)
    # num_of_classes = len(np.unique(y))
    # print(num_of_classes)
    # y = keras.utils.to_categorical(y, num_classes=num_of_classes)

    # f = open("/home/mia/drone-classification/03-28-23/run7.log", "w")
    # loss_list, acc_list, time_list = [], [], []
    # for i in range(1, 11):
    #     # training and testing set split
    #     x_train, x_test, y_train, y_test = train_test_split(
    #         x, y, test_size=0.2, random_state=42
    #     )

    #     print(x_train.shape[0])
    #     print(x_train.shape[1])
    #     print(x_train.shape)

    #     # load model
    #     model = modelloader.get_model(
    #         "cnn", input_shape=x_train.shape[1], num_class=num_of_classes
    #     )
    #     keras.utils.plot_model(
    #         model,
    #         to_file="/home/mia/drone-classification/figures/keras-model.png",
    #         show_shapes=True,
    #         show_layer_names=False,
    #         rankdir="TB",
    #         dpi=200,
    #     )

    #     f.write("\nRun " + str(i) + "\n")
    #     start_time = time.time()
    #     history = model.fit(x_train, y_train, validation_split=0.2, epochs=100)
        
    #     # history = training(model, x_train, y_train, epochs=100)
    #     end_time = time.time()
    #     time_list.append(end_time - start_time)
    #     f.write("Total training time: " + str(end_time - start_time) + "\n")
    #     print("Total training time:", end_time - start_time)

    #     # model.save("models/model.h5")
    #     y_pred = model.predict(x_test)
    #     y_pred = np.around(y_pred)
    #     cal_confusion_matrix(y_test, y_pred, f)
    #     plot_history(history)
        
    #     loss, acc = model.evaluate(x_test, y_test)
    #     print("Test loss:", loss)
    #     print("Test accuracy:", acc)
    #     f.write("Test loss: " + str(loss) + "\n")
    #     f.write("Test accuracy: " + str(acc) + "\n")
    #     loss_list.append(loss)
    #     acc_list.append(acc)
    # acc_mean = np.mean(acc_list)
    # acc_std = np.std(acc_list)
    # loss_mean = np.mean(loss_list)
    # loss_std = np.std(loss_list)
    # time_mean = np.mean(time_list)
    # time_std = np.std(time_list)
    # res_str = (
    #     "\nAccuracy average: "
    #     + str(acc_mean)
    #     + ", STD: "
    #     + str(acc_std)
    #     + "\nLoss average: "
    #     + str(loss_mean)
    #     + ", STD: "
    #     + str(loss_std)
    #     + "\nTraining time average: "
    #     + str(time_mean)
    #     + ", STD: "
    #     + str(time_std)
    # )

    
    # f.write(res_str)
    # print(res_str)
    # f.close()

    

