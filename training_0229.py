import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from yellowbrick.classifier import ROCAUC
from sklearn.preprocessing import LabelBinarizer

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

warnings.filterwarnings("ignore")

# This is a test!

# calculate confusion matrix
def training(model, training_data, training_labels, epochs=100, **kwargs):
    ################################################################
    # train model
    ################################################################
    # 1. loss function
    # 2. weighted categories?
    # 3. data augmentation methods?
    # 4. optimizer options: Adam, SGD, RMSProp, etc.

    # for CNN
    training_history = model.fit(
        training_data, training_labels, epochs=epochs, validation_data=(x_test, y_test)
    )

    # for RNN

    # training_history = model.fit(
    #     training_data,
    #     training_labels,
    #     batch_size=32,
    #     epochs=epochs,
    #     validation_data=(x_test, y_test),
    # )
    # for KNN
    # training_history = model.fit(training_data, training_labels)

    # 5. save model
    # 6. save training configurations
    # 7. training history plots
    # summarize history for accuracy

    # 8. training time
    ################################################################

    ################################################################
    # generate testing statistics
    ################################################################
    # 1. get test set prediction results from trained model
    # 2. save prediction results in CSV?
    # 3. calculate confusion matrix
    # 4. ROC & AUC curve?

    ################################################################
    # save everything as much as possible
    ################################################################
    return training_history


def plot_history(training_history):
    # print("This is a test...")
    # list all data in history
    plt.figure()
    # plt.figure(figsize=(5,3))
    plt.plot(training_history.history["loss"])
    plt.plot(training_history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper right")
    plt.tight_layout()
    plt.savefig("/home/mia/drone-classification/figures/loss-history.png", dpi=200)

    plt.figure()
    # plt.figure(figsize=(5,3))
    plt.plot(training_history.history["accuracy"])
    plt.plot(training_history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="lower right")
    plt.tight_layout()
    plt.savefig("/home/mia/drone-classification/figures/acc-history.png", dpi=200)



def cal_confusion_matrix(y_test, y_pred, f):
    print("number of tests: ", len(y_test))
    print("Confusion Matrix: ")
    # mcm = multilabel_confusion_matrix(y_test, y_pred)

    lables = ['David_Tricopter','Matrice200','Matrice200_V2','Mavic_Air2', 'Mavic_Mini1',
              'Mavic_Mini2', 'Mavic2pro', 'Mavic2s', 'Phantom2','Phantom4', 
              'Tello_TT', 'EvoII', 'Hasakee_Q11', 'PhenoBee', 'Splash3_plus',
              'X5SW', 'X5UW', 'X20', 'X20P', 'X26', 'UDIU46', 'Yuneec',
              ]  
    mcm = confusion_matrix(np.asarray(y_test).argmax(axis=1), np.asarray(y_pred).argmax(axis=1))
    print(mcm)
    

    sns.set(rc={'figure.figsize':(18, 15)})
    ax= plt.subplot()
    sns.heatmap(mcm, annot=True, fmt='g', ax=ax)
    

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(lables, rotation=90, ha='right'); ax.yaxis.set_ticklabels(lables, rotation=0, ha='right')
    
    # plt.show()
    plt.savefig('/home/mia/drone-classification/figures/cm040101.png', dpi=200)


    report_str = classification_report(
        y_test,
        y_pred,
        target_names=[
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
            "20",
            "21",
        ],
    )
    print(report_str)
    acc_str = str(accuracy_score(y_test, y_pred))
    print("Accuracy:", acc_str)

    y_str = ""

    log_str = "\n" + report_str + "\nAccuracy: " + acc_str + "\n" + y_str
    f.write(log_str)



def plot_ROC_curve(model, x_train, y_train, x_test, y_test):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder={0: 'functional', 
                                        1: 'needs repair', 
                                        2: 'nonfunctional'})
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(x_train, y_train)
    visualizer.score(x_test, y_test)
    visualizer.show()
    
    return visualizer

if __name__ == "__main__":
    # load data and apply preprocessing method
    x, y = dataloader.load_dataset(
        dataset="Drone_Dataset/0308_22_classes", feature="mfcc"
    )

    print(y)
    num_of_classes = len(np.unique(y))
    print(num_of_classes)
    y = keras.utils.to_categorical(y, num_classes=num_of_classes)

    f = open("/home/mia/drone-classification/04-01-23/run4.log", "w")
    loss_list, acc_list, time_list = [], [], []
    for i in range(1, 2):
        # training and testing set split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        # print(x_train.shape[0])
        # print(x_train.shape[1])
        # print(x_train.shape)

        # load model
        model = modelloader.get_model(
            "cnn", input_shape=x_train.shape[1], num_class=num_of_classes
        )
        keras.utils.plot_model(
            model,
            to_file="/home/mia/drone-classification/figures/keras-model.png",
            show_shapes=True,
            show_layer_names=False,
            rankdir="TB",
            dpi=200,
        )

        f.write("\nRun " + str(i) + "\n")
        start_time = time.time()
        history = model.fit(x_train, y_train, validation_split=0.2, epochs=100)
    
        # history = training(model, x_train, y_train, epochs=100)
        end_time = time.time()
        time_list.append(end_time - start_time)
        f.write("Total training time: " + str(end_time - start_time) + "\n")
        print("Total training time:", end_time - start_time)

        # model.save("models/model.h5")
        y_pred = model.predict(x_test)
        y_pred = np.around(y_pred)
        cal_confusion_matrix(y_test, y_pred, f)
        plot_history(history)
        
        loss, acc = model.evaluate(x_test, y_test)
        print("Test loss:", loss)
        print("Test accuracy:", acc)
        f.write("Test loss: " + str(loss) + "\n")
        f.write("Test accuracy: " + str(acc) + "\n")
        loss_list.append(loss)
        acc_list.append(acc)
    acc_mean = np.mean(acc_list)
    acc_std = np.std(acc_list)
    loss_mean = np.mean(loss_list)
    loss_std = np.std(loss_list)
    time_mean = np.mean(time_list)
    time_std = np.std(time_list)
    res_str = (
        "\nAccuracy average: "
        + str(acc_mean)
        + ", STD: "
        + str(acc_std)
        + "\nLoss average: "
        + str(loss_mean)
        + ", STD: "
        + str(loss_std)
        + "\nTraining time average: "
        + str(time_mean)
        + ", STD: "
        + str(time_std)
    )

    
    f.write(res_str)
    print(res_str)
    f.close()

    

