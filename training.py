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

warnings.filterwarnings("ignore")


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
    # training_history = model.fit(
    #     training_data, training_labels, epochs=epochs, validation_data=(X_test, y_test)
    # )

    # for RNN
    training_history = model.fit(
        training_data,
        training_labels,
        batch_size=32,
        epochs=epochs,
        validation_data=(x_test, y_test),
    )
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
    # list all data in history
    plt.figure()
    plt.plot(training_history.history["accuracy"])
    plt.plot(training_history.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="lower right")
    plt.tight_layout()
    plt.savefig("figures/acc-history.png", dpi=200)

    plt.figure()
    plt.plot(training_history.history["loss"])
    plt.plot(training_history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "test"], loc="upper right")
    plt.tight_layout()
    plt.savefig("figures/loss-history.png", dpi=200)


def cal_confusion_matrix(y_test, y_pred, f):
    print("number of tests: ", len(y_test))
    print("Confusion Matrix: ")
    mcm = multilabel_confusion_matrix(y_test, y_pred)
    print(mcm)
    f.write(str(mcm))
    # print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
    # # print('Recall: %.3f' % recall_score(y_test, y_pred, pos_label='drones'))
    # # print('Precision: %.3f' % precision_score(y_test, y_pred, pos_label='drones'))
    # # print('F1 Score: %.3f' % f1_score(y_test, y_pred, pos_label='drones'))
    # print('Recall: %.3f' % recall_score(y_test, y_pred))
    # print('Precision: %.3f' % precision_score(y_test, y_pred))
    # print('F1 Score: %.3f' % f1_score(y_test, y_pred))

    # print("***********************")
    # precision, recall, fscore, support = score(y_test, y_pred)

    # print("precision:", precision)
    # print("recall:", recall)
    # print("fscore:", fscore)
    # print("support:", support)

    report_str = classification_report(
        y_test, y_pred, target_names=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    )
    print(report_str)

    acc_str = str(accuracy_score(y_test, y_pred))
    print("Accuracy:", acc_str)

    y_str = ""
    # for y_p, y_t in zip(y_pred, y_test):
    #     y_str += "Pred: ["
    #     for p in y_p:
    #         y_str = y_str + str(p) + ","
    #     y_str = y_str + "]\n"
    #     y_str += "True: ["
    #     for t in y_t:
    #         y_str = y_str + str(t) + ","
    #     y_str = y_str + "]\n\n"

    log_str = "\n" + report_str + "\nAccuracy: " + acc_str + "\n" + y_str

    f.write(log_str)


if __name__ == "__main__":
    # training set statistic summary

    # load data and apply preprocessing method
    # x, y = dataloader.load_dataset(dataset="aira-uas", feature="stft_chroma")
    # x, y = dataloader.load_dataset(dataset="droneaudiodataset/binary_drone_audio", feature="mel")
    x, y = dataloader.load_dataset(
        dataset="Drone_Dataset/0823_10_classes", feature="mfcc"
    )

    num_of_classes = len(np.unique(y))
    y = keras.utils.to_categorical(y, num_classes=num_of_classes)

    f = open("multi-run-batch32-epoch100.log", "w")
    loss_list, acc_list, time_list = [], [], []
    for i in range(1, 11):
        # training and testing set split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42
        )

        # load model
        model = modelloader.get_model(
            "cnn", input_shape=x_train.shape[1], num_class=num_of_classes
        )
        keras.utils.plot_model(
            model,
            to_file="figures/keras-model.png",
            show_shapes=True,
            show_layer_names=False,
            rankdir="TB",
            dpi=200,
        )

        f.write("\nRun " + str(i) + "\n")
        start_time = time.time()
        history = training(model, x_train, y_train, epochs=100)
        end_time = time.time()
        time_list.append(end_time - start_time)
        f.write("Total training time: " + str(end_time - start_time) + "\n")
        print("Total training time:", end_time - start_time)
        plot_history(history)

        # model.save("models/model.h5")
        y_pred = model.predict(x_test)
        y_pred = np.around(y_pred)
        cal_confusion_matrix(y_test, y_pred, f)

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
