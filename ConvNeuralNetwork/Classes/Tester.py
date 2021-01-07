import torch
from ConvNeuralNetwork.Classes.PImages import PImages
import numpy as np

from ConvNeuralNetwork.Classes.Performance import write_testing_performance


class Tester():
    def __init__(self, test_loader, criterion, net, plot_data, features):
        self.test_loader = test_loader
        self.criterion = criterion
        self.net = net
        self.plot_data = plot_data
        self.features = features

    def test(self):
        device = ("cuda" if torch.cuda.is_available() else "cpu")
        correct = 0
        total = 0

        label_list = np.array(list())
        predicted_list = np.array(list())

        for i, (inputs, labels) in enumerate(self.test_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs.to(device)
            labels.to(device)

            # forward + backward + optimize
            outputs = self.net(inputs)

            # calculate loss
            loss = self.criterion(outputs, labels).item()

            # calculate accuracy
            print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            print(predicted)
            print(labels)
            correct += (predicted == labels).sum().item()

            label_list = np.append(label_list, labels.numpy())
            predicted_list = np.append(predicted_list, predicted.numpy())
            print(predicted_list)

        if self.plot_data:
            PImages(self.test_loader).plot()

        accuracy = 100 * correct / total

        write_testing_performance(loss, accuracy, label_list, predicted_list, self.features)

        print('Finished Testing')
