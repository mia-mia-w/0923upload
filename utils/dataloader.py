import os
import numpy as np
import librosa
import soundfile as sf


def load_dataset(dataset: str, feature: str = None):
    """
    Load the dataset from files with corresponding labels and return
    >>> X, y = load_dataset(dataset="AIRA-UAS", feature="mfcc")
    :param dataset: The name of the dataset
    :param feature: The feature used for data preprocessing (feature extraction)
    :return:
        ndarray: Processed data in a numpy array
        ndarray: Corresponding labels in a 1D numpy array
    """

    X, y = [], []
    ################################################################
    # load data from files
    ################################################################
    if dataset == "AIRA-UAS":
        parent_dir = "/home/mia/drone-classification/datasets/AIRA-UAS"
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop AIRA-UAS
            protocols_dir = os.path.join(parent_dir, sub_dir)
            if os.path.isdir(protocols_dir):
                for sub_sub_dir in os.listdir(protocols_dir):
                    # for loop Protocol
                    recording_dir = os.path.join(protocols_dir, sub_sub_dir)
                    if os.path.isdir(recording_dir):
                        for fn in os.listdir(recording_dir):
                            # for loop recording
                            fn = os.path.join(recording_dir, fn)
                            data, samplerate = sf.read(fn, dtype="float32")
                            X.append(data)
                            y.append(sub_dir)

    elif dataset == "DroneAudioDataset/Binary_Drone_Audio":
        parent_dir = "/home/mia/drone-classification/datasets/DroneAudioDataset/Binary_Drone_Audio"
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop Binary_Drone_Audio
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "drones":
                        y.append(1)
                    elif sub_dir == "noise":
                        y.append(0)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/3s_DJI":
        parent_dir = "/home/mia/drone-classification/datasets/Drone_Dataset/3s_DJI"
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    data, samplerate = librosa.load(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200_3s":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro_3s":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4_3s":
                        y.append(2)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/5s_DJI":
        parent_dir = "/home/mia/drone-classification/datasets/Drone_Dataset/5s_DJI"
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4_5s":
                        y.append(2)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/10s_DJI":
        parent_dir = "/home/mia/drone-classification/datasets/Drone_Dataset/10s_DJI"
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200_10s":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro_10s":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4_10s":
                        y.append(2)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/DJI+Evo":
        parent_dir = "/home/mia/drone-classification/datasets/Drone_Dataset/DJI+Evo"
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200_5s":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro_5s":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4_5s":
                        y.append(2)
                    elif sub_dir == "EvoII_5s":
                        y.append(3)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/DJI_4_classes":
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/DJI_4_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(2)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(3)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/Syma":
        index = 0
        parent_dir = "/home/mia/drone-classification/datasets/Drone_Dataset/Syma"
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    print(str(index) + ": " + fn)
                    index += 1
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "Syma_X5UW":
                        y.append(0)
                    elif sub_dir == "Syma_X5SW":
                        y.append(1)
                    elif sub_dir == "Syma_X8SW":
                        y.append(2)
                    elif sub_dir == "Syma_X20":
                        y.append(3)
                    elif sub_dir == "Syma_X20P":
                        y.append(4)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/0823_11_classes":
        index = 0
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/0823_11_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)
            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    print(str(index) + ": " + fn)
                    index += 1
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200":
                        y.append(0)
                    elif sub_dir == "DJI_Matrice200_V2":
                        y.append(1)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(2)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(3)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(4)
                    elif sub_dir == "EvoII":
                        y.append(5)
                    elif sub_dir == "Syma_X5SW":
                        y.append(6)
                    elif sub_dir == "Syma_X5UW":
                        y.append(7)
                    elif sub_dir == "Syma_X20":
                        y.append(8)
                    elif sub_dir == "Syma_X20P":
                        y.append(9)
                    elif sub_dir == "Yuneec":
                        y.append(10)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/0823_10_classes":
        index = 0
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/0823_10_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)
            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    print(str(index) + ": " + fn)
                    index += 1
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200":
                        y.append(0)
                    elif sub_dir == "DJI_Matrice200_V2":
                        y.append(1)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(2)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(3)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(4)
                    elif sub_dir == "EvoII":
                        y.append(5)
                    elif sub_dir == "Syma_X5SW":
                        y.append(6)
                    elif sub_dir == "Syma_X5UW":
                        y.append(7)
                    elif sub_dir == "Syma_X20P":
                        y.append(8)
                    elif sub_dir == "Yuneec":
                        y.append(9)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/1006_15_classes":
        index = 0
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/1006_15_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)
            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    print(str(index) + ": " + fn)
                    index += 1
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "David_Tricopter":
                        y.append(0)
                    elif sub_dir == "DJI_Matrice200":
                        y.append(1)
                    elif sub_dir == "DJI_Matrice200_V2":
                        y.append(2)
                    elif sub_dir == "DJI_Mavic_Air2":
                        y.append(3)
                    elif sub_dir == "DJI_Mavic_Mini1":
                        y.append(4)
                    elif sub_dir == "DJI_Mavic_Mini2":
                        y.append(5)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(6)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(7)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(8)
                    elif sub_dir == "EvoII":
                        y.append(9)
                    elif sub_dir == "Syma_X5SW":
                        y.append(10)
                    elif sub_dir == "Syma_X5UW":
                        y.append(11)
                    elif sub_dir == "Syma_X20":
                        y.append(12)
                    elif sub_dir == "Syma_X20P":
                        y.append(13)
                    elif sub_dir == "Yuneec":
                        y.append(14)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/1006_12_classes":
        index = 0
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/1006_12_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)
            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    print(str(index) + ": " + fn)
                    index += 1
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "David_Tricopter":
                        y.append(0)
                    elif sub_dir == "DJI_Matrice200":
                        y.append(1)
                    elif sub_dir == "DJI_Matrice200_V2":
                        y.append(2)
                    elif sub_dir == "DJI_Mavic_Air2":
                        y.append(3)
                    elif sub_dir == "DJI_Mavic_Mini1":
                        y.append(4)
                    elif sub_dir == "DJI_Mavic_Mini2":
                        y.append(5)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(6)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(7)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(8)
                    elif sub_dir == "EvoII":
                        y.append(9)
                    elif sub_dir == "Syma_X20P":
                        y.append(10)
                    elif sub_dir == "Yuneec":
                        y.append(11)
                    # y.append(sub_dir)
        print("Reading finished.")

    
    elif dataset == "Drone_Dataset/0709_9_classes":
        index = 0
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/0709_9_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)
            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    print(str(index) + ": " + fn)
                    index += 1
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(2)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(3)
                    elif sub_dir == "EvoII":
                        y.append(4)
                    elif sub_dir == "Syma_X5SW":
                        y.append(5)
                    elif sub_dir == "Syma_X8SW":
                        y.append(6)
                    elif sub_dir == "Syma_X20P":
                        y.append(7)
                    elif sub_dir == "Yuneec":
                        y.append(8)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/0709_8_classes":
        index = 0
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/0709_8_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)
            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    # print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    print(str(index) + ": " + fn)
                    index += 1
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(2)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(3)
                    elif sub_dir == "EvoII":
                        y.append(4)
                    elif sub_dir == "Syma_X5SW":
                        y.append(5)
                    elif sub_dir == "Syma_X20P":
                        y.append(6)
                    elif sub_dir == "Yuneec":
                        y.append(7)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/0709_7_classes":
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/0709_7_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(2)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(3)
                    elif sub_dir == "Syma_X5SW":
                        y.append(4)
                    elif sub_dir == "Syma_X8SW":
                        y.append(5)
                    elif sub_dir == "Syma_X20P":
                        y.append(6)
                    # y.append(sub_dir)
        print("Reading finished.")

    elif dataset == "Drone_Dataset/0709_6_classes":
        parent_dir = (
            "/home/mia/drone-classification/datasets/Drone_Dataset/0709_6_classes"
        )
        sub_dirs = os.listdir(parent_dir)
        for sub_dir in sub_dirs:  # for loop 3s_DJI
            drone_audio_dir = os.path.join(parent_dir, sub_dir)

            if os.path.isdir(drone_audio_dir):
                # for sub_sub_dir in os.listdir(drone_audio_dir):
                #     # for loop binary_drone_audio & multiclass_drone_audio
                #     recording_dir = os.path.join(drone_audio_dir, sub_sub_dir)
                #     if os.path.isdir(recording_dir):
                for fn in os.listdir(drone_audio_dir):
                    print(sub_dir)
                    fn = os.path.join(drone_audio_dir, fn)
                    data, samplerate = sf.read(fn, dtype="float32")
                    X.append(data)
                    if sub_dir == "DJI_Matrice200":
                        y.append(0)
                    elif sub_dir == "DJI_Mavic2pro":
                        y.append(1)
                    elif sub_dir == "DJI_Phantom4":
                        y.append(2)
                    elif sub_dir == "DJI_Phantom2":
                        y.append(3)
                    elif sub_dir == "EvoII":
                        y.append(4)
                    elif sub_dir == "Yuneec":
                        y.append(5)

                    # y.append(sub_dir)
        print("Reading finished.")

    else:
        raise KeyError("Dataset name not found: " + dataset)

    ################################################################
    # apply preprocessing, or feature extraction methods to the data
    ################################################################
    if feature is None:
        return X, y
    elif feature == "mfcc":
        X = mfcc_feature(X)
    elif feature == "stft_chroma":
        X = stft_chroma_feature(X)
    elif feature == "mel":
        X = mel_feature(X)
    elif feature == "contrast":
        X = contrast_feature(X)
    return X, y


def mfcc_feature(X):
    # extract mfcc features from data
    mfcc = []
    for index, x in enumerate(X):
        mfcc.append(np.mean(librosa.feature.mfcc(x, n_mfcc=128).T, axis=0))
        # mfcc = librosa.feature.mfcc(x)
        print(str(index) + "/" + str(len(X)))
    print(len(mfcc))
    print("Feature extraction done.")
    return np.array(mfcc)


def contrast_feature(X):
    # extract stft features from data
    contrast = []
    for x in X:
        contrast.append(np.mean(librosa.feature.spectral_contrast(x).T, axis=0))
    print(len(contrast))
    return np.array(contrast)


def stft_chroma_feature(X):
    # extract stft features from data
    stft = []
    for x in X:
        temp = np.abs(librosa.stft(x))
        stft.append(np.mean(librosa.feature.chroma_stft(S=temp).T, axis=0))
    print(len(stft))
    return np.array(stft)


def mel_feature(X):
    mel = []
    for index, x in enumerate(X):
        # melspectrogram
        # (300, 128)
        mel.append(np.mean(librosa.feature.melspectrogram(x).T, axis=0))
        # mel.append(librosa.feature.melspectrogram(x))
        print(str(index) + "/" + str(len(X)))
    print(len(mel))
    return np.array(mel)
