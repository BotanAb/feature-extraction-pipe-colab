import matplotlib as plt
import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import os


LOG_PATH = './Log/'
PERF_LOG_FILE = 'performance_statistics'


def write_training_performance(trainLoss, trainAccuracy, lbllist, predlist, features):
    logFile = PERF_LOG_FILE
    init_log(logFile)

    log_text(logFile, "---Training performance---")

    write_training_loss_accuracy(trainLoss, trainAccuracy, logFile)
    plot_confusion_matrix(lbllist, predlist, features, 'train')


def write_testing_performance(testLoss, testAccuracy, lbllist, predlist, features):
    logFile = PERF_LOG_FILE
    log_text(logFile, "---Testing performance---")

    print("Loss of the network on the test images: " + str(round(testLoss, 2)))
    print("Accuracy of the network on the test images: %d %%" % testAccuracy)
    log_text(logFile, 'Loss: ' + str(round(testLoss, 2)))
    log_text(logFile, 'Accuracy: ' + str(round(testAccuracy, 2)) + '%')
    plot_confusion_matrix(lbllist, predlist, features, 'test')


def write_training_loss_accuracy(loss, accuracy, logFile):
    data_frame = pd.DataFrame()
    data_frame['Epoch'] = list(range(1, len(loss) + 1))
    data_frame['Loss'] = np.round(loss, 2)
    data_frame['Accuracy'] = np.round(accuracy, 2)

    data_frame.plot(x='Epoch', y=["Loss", "Accuracy"])
    # plt.xticks(rotation=45)
    plt.title("Loss and accuracy performance training")
    plt.xlabel("Epoch")
    plt.ylabel("Performance (batch)")

    # plt.show()
    plt.savefig(LOG_PATH + 'training_loss_and_accuracy.png')
    plt.close()

    log_data_frame(logFile, data_frame)


def plot_confusion_matrix(label_list, prediction_list, classes, type):
    conf_matrix = confusion_matrix(label_list, prediction_list)

    # print(confusionMatrix)

    plt.imshow(conf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
    if type == 'train':
        plt.title("Confusion matrix training")
    elif type == 'test':
        plt.title("Confusion matrix testing")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = "d"
    thresh = conf_matrix.max() / 2.
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    # plt.show()
    if type == 'train':
        plt.savefig(LOG_PATH + 'confusion_matrix_training.png')
    elif type == 'test':
        plt.savefig(LOG_PATH + 'confusion_matrix_testing.png')
    plt.close()


def log_data_frame(name, df):
    df.to_csv(LOG_PATH + name + '.txt', sep='\t', index=False, mode='a')
    log_text(name, '')


def log_text(name, text):
    f = open(LOG_PATH + name + '.txt', 'a')
    f.write(text + '\n')


def init_log(name):
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
    open(LOG_PATH + name + '.txt', 'w')