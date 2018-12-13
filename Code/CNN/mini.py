import cv2
import numpy as np
import os
from cnn import LeNet
from sklearn.model_selection import train_test_split
import random
from keras.optimizers import SGD, Adam
from scipy import stats
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import argparse
import sys

parser = argparse.ArgumentParser(description='Run CNN on given dataset')
parser.add_argument('--dataset_folder_path', metavar='N', type=str, help='Path to a folder which contains data.npy and labels.npy')

genres =	{
  "metal": 0,
  "pop": 1,
  "disco": 2,
  "blues":3,
  "classical":4,
  "reggae":5,
  "rock":6,
  "hiphop":7,
  "country":8,
  "jazz":9
}

def construct_input_matrix(input_loc, input_h, input_w, channels, no_of_imgs):
    a = np.zeros(shape=(no_of_imgs,input_h,input_w, channels))
    labels = np.zeros(shape=(no_of_imgs,1))
    index = 0
    root_to_save = input_loc
    for root, dirs, files in os.walk(root_to_save):
        for directory in dirs:
            for root1, dirs1, files1 in os.walk(os.path.join(root, directory)):
                for f in files1:
                    if directory in genres:
                        labels[index] = genres[directory]
                    img = cv2.imread(os.path.join(root1,f),cv2.IMREAD_COLOR) 
                    a[index,:,:,:] = img
                    index += 1
    a = stats.zscore(a)
    a = np.nan_to_num(a)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/data', a)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/labels', labels)
    return a, labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def construct_input_matrix_grayscale(input_loc, input_h, input_w, no_of_imgs):
    a = np.zeros(shape=(no_of_imgs,input_h,input_w))
    labels = np.zeros(shape=(no_of_imgs,1))
    index = 0
    root_to_save = input_loc
    for root, dirs, files in os.walk(root_to_save):
        for directory in dirs:
            for root1, dirs1, files1 in os.walk(os.path.join(root, directory)):
                for f in files1:
                    if directory in genres:
                        labels[index] = genres[directory]
                    img = cv2.imread(os.path.join(root1,f),cv2.IMREAD_GRAYSCALE) 
                    a[index,:,:] = img
                    index += 1
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/data', a)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/labels', labels)
    return a, labels

def main(folder_path):
    data_path = os.path.join(folder_path, "toy_data.npy")
    label_path = os.path.join(folder_path, "toy_label.npy")
    data = np.load(data_path)
    labels = np.load(label_path)
    train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.01)
    model = LeNet.build_model_lenet(324,300,3,10)
    opt = Adam(lr=0.001)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) #categorical_crossentropy
    print("[INFO] training...")
    model.fit(train_data, train_label, batch_size=64, epochs=10, verbose=1, validation_data=(test_data, test_label) )
    (loss, accuracy) = model.evaluate(test_data, test_label, batch_size=64, verbose=1)
    results = model.predict_classes(test_data, batch_size=64, verbose =1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
    print("loss: {}".format(loss))
    '''Uncomment if you want to plot confusion matrix
    res = confusion_matrix(test_label, results, labels=[0,1,2,3,4,5,6,7,8,9])#"metal", "pop", "disco", "blues", "classical", "reggae", "rock", "hiphop", "country", "jazz"]) 
    class_names = [0,1,2,3,4,5,6,7,8,9]
    plot_confusion_matrix(res, class_names)
    plt.show()
    '''

if __name__ == "__main__":
    flags = parser.parse_args(sys.argv[1:])
    main(flags.dataset_folder_path)