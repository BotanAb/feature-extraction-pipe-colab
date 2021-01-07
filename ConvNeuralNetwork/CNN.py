import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import argparse
from pathlib import Path
from torchvision import datasets, models, transforms
from ConvNeuralNetwork.Classes.NetModel import NetModel
from ConvNeuralNetwork.Classes.Trainer import Trainer
from ConvNeuralNetwork.Classes.Tester import Tester
from ConvNeuralNetwork.Classes.MyDataset import load_dataset
from ImagePrep.BoundingBox import prep_bounding_boxes
from ImagePrep.BoundingBox import renderImageWithBoundingBox
from ImagePrep.BoundingBox import getImageFromBoundingBox

from ConvNeuralNetwork.Classes.MonitorCNN import Animate
import multiprocessing
import glob

import os

from PIL import Image
from torch.autograd import Variable


class CNN:
    def __init__(self, features):
        self.features = features
        self.loadPretrainedModel = False
        self.plot_data = False
        self.train_CNN = False

        self.transform = transforms.Compose(
            [
                transforms.Resize((30, 30)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def train_cnn(self):
        ######################################### load pre trained model ##############################################

        if self.loadPretrainedModel:
            net = self.define_pre_trained_cnn()
            self.show_convolutional_layers(net)
        else:
            net = self.define_cnn()

        plot_data, train_cnn, demo_mode = self.adjust_workflow()
        train_loader, test_loader, classes = self.load_dataset(self.transform, batch_size=3, shuffle=True,
                                                               pin_memory=True,
                                                               num_workers=0)
        criterion, optimizer = self.loss_function(net, learning_rate=0.001, momentum=0.9)

        if train_cnn:

            num_epochs = 25

            if demo_mode:
                animated_plot = Animate()
                p2 = multiprocessing.Process(target=animated_plot.start)
                p1 = multiprocessing.Process(
                    target=Trainer(train_loader, optimizer, criterion, net, num_epochs,
                                   demo_mode, self.features).train, )

                net.train()

                p2.start()
                print("animate.start")
                p1.start()
                print("train.start")
                p1.join()
                p2.terminate()

            else:
                net.train()
                Trainer(train_loader, optimizer, criterion, net, num_epochs,
                        demo_mode, self.features).train()

            path = self.save_cnn(train_cnn, net)
            Tester(test_loader, criterion, net, plot_data, self.features).test()
            self.restore_cnn(path, self.features)

    def adjust_workflow(self):
        plot_data = False
        train_cnn = True
        demo_mode = False
        return plot_data, train_cnn, demo_mode

    def load_dataset(self, transform, batch_size, shuffle, pin_memory, num_workers):
        train_loader, test_loader, classes = load_dataset(transform, batch_size, shuffle, pin_memory, num_workers,
                                                          self.features)
        return train_loader, test_loader, classes

    def define_cnn(self):
        net = NetModel(self.features)
        return net

    def show_convolutional_layers(self, model):
        conv_layers = []
        model_weights = []
        model_children = list(model.children())
        counter = 0
        # append all the conv layers and their respective weights to the list
        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print(f"Total convolutional layers: {counter}")

        for weight, conv in zip(model_weights, conv_layers):
            # print(f"WEIGHT: {weight} \nSHAPE: {weight.shape}")
            print(f"CONV: {conv} ====> SHAPE: {weight.shape}")

        self.visualize_convolutional_layers(model_weights)
        for feature in self.features:
            feature_path = glob.glob('./DatasetImages/' + feature + '*.jpg', recursive=True)
            img = self.read_image(feature_path[0])
            img = self.define_transforms(img)
            print(conv_layers)
            self.pass_image_and_visualize(conv_layers, img, feature)

    def read_image(self, feature_path):
        # read and visualize an image
        img = cv.imread(feature_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        plt.imshow(img)
        # plt.show()
        return img

    def define_transforms(self, img):
        # define the transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])
        img = np.array(img)
        # apply the transforms
        img = transform(img)
        print(img.size())
        # unsqueeze to add a batch dimension
        img = img.unsqueeze(0)
        print(img.size())
        return img

    def pass_image_and_visualize(self, conv_layers, img, feature):
        # pass the image through all the layers
        results = [conv_layers[0](img)]
        for i in range(1, len(conv_layers)):
            # pass the result from the last layer to the next layer
            results.append(conv_layers[i](results[-1]))
        # make a copy of the `results`
        outputs = results

        # visualize 64 features from each layer
        # (although there are more feature maps in the upper layers)
        for num_layer in range(len(outputs)):
            plt.figure(figsize=(30, 30))
            layer_viz = outputs[num_layer][0, :, :, :]
            layer_viz = layer_viz.data
            print(layer_viz.size())
            for i, filter in enumerate(layer_viz):
                if i == 64:  # we will visualize only 8x8 blocks from each layer
                    break
                plt.subplot(8, 8, i + 1)
                plt.imshow(filter, cmap='gray')
                plt.axis("off")
            print(f"Saving layer {num_layer} feature maps...")
            plt.savefig(f"./ConvNeuralNetwork/outputs/{feature}_layer_{num_layer}.png")
            # plt.show()
            plt.close()

    def visualize_convolutional_layers(self, model_weights):
        plt.figure(figsize=(20, 17))
        for j, model_weight in enumerate(model_weights):
            for i, filter in enumerate(model_weight):
                plt.subplot(8, 8,
                            i + 1)  # (8, 8) because in conv0 we have 7x7 filters and total of 64 (see printed shapes)
                plt.imshow(filter[0, :, :].detach(), cmap='gray')
                plt.axis('off')
                plt.savefig('./ConvNeuralNetwork/filters/filtersLayer' + str(j) + '.png')
            # plt.show()

    def define_pre_trained_cnn(self):
        net = NetModel(self.features)
        net.load_state_dict(torch.load('./ConvNeuralNetwork/savedCNN/transfer_test.pth'))
        # net.fc3 = torch.nn.Linear(net.fc3.in_features, 4)
        return net

    def loss_function(self, net, learning_rate, momentum):
        import torch.optim as optim
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
        return criterion, optimizer

    def save_cnn(self, train_cnn, net):
        cnn_path = './ConvNeuralNetwork/savedCNN/'
        cnn_name = 'transfer_test.pth'

        if not os.path.exists(cnn_path):
            os.makedirs(cnn_path)

        if train_cnn:
            torch.save(net.state_dict(), cnn_path + cnn_name)
        return cnn_path + cnn_name

    def restore_cnn(self, PATH, features):
        net = NetModel(features)
        net.load_state_dict(torch.load(PATH))
        return net

    def predict(self, imagePath, features):
        ############ Open Image and convert to Tensor#####################

        image = Image.open(imagePath)
        image = self.transform(image)
        image = image.unsqueeze(0)

        ############ Load (trained) NN ###################################
        net = self.restore_cnn(PATH='./ConvNeuralNetwork/savedCNN/transfer_test.pth', features=features)
        ############ Predict Feature of Image ############################
        outputs = net(image)
        _, predicted = torch.max(outputs.data, 1)
        return features[predicted.item()]

    def predict_multiple(self, image_path, features):
        predicted_items = []

        bounding_boxes = prep_bounding_boxes(image_path)
        print(bounding_boxes)
        for bounding_box in bounding_boxes:
            ############ Open Image and convert to Tensor#####################
            image = getImageFromBoundingBox(image_path, bounding_box)
            image = self.transform(image)
            image = image.unsqueeze(0)

            ############ Load (trained) NN ###################################
            net = self.restore_cnn(PATH='./ConvNeuralNetwork/savedCNN/transfer_test.pth', features=features)

            ############ Predict Feature of Image ############################
            outputs = net(image)
            # print(outputs.data)
            _, predicted = torch.max(outputs.data, 1)
            predicted_items.append(features[predicted.item()])

        print(predicted_items)

        renderImageWithBoundingBox(image_path, bounding_boxes, predicted_items)


if __name__ == '__main__':
    features = ["fist", "palm", "thumb"]
    test = CNN(features)
    test.train_cnn()
