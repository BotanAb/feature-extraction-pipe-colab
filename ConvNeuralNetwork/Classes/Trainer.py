import torch
import numpy as np

from ConvNeuralNetwork.Classes.Performance import write_training_performance


class Trainer():
    def __init__(self, train_loader, optimizer, criterion, net, num_epochs, demo_mode, features):
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.net = net
        self.num_epochs = num_epochs
        self.demo_mode = demo_mode
        self.features = features

    def train(self):
        # device = ("cuda" if torch.cuda.is_available() else "cpu")
        train_loss = []
        train_accuracy = []
        device = "cpu"
        # print("Device = " + device)

        n = 0
        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            epoch_loss = 0.0

            correct = 0
            total = 0

            label_list = np.array(list())
            predicted_list = np.array(list())

            for i, (inputs, labels) in enumerate(self.train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs.to(device)
                labels.to(device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)

                # calculate loss
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print loss statistics
                running_loss += loss.item()
                epoch_loss += loss.item()

                # calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                label_list = np.append(label_list, labels.numpy())
                predicted_list = np.append(predicted_list, predicted.numpy())

                n += 1
                if n % 100 == 0:  # print every 2000 mini-batches
                    print('[epoch: %d, Batch: %5d] loss: %.3f' % (epoch + 1, n + 1, running_loss / 2))
                    if self.demo_mode:
                        log_loss(str(epoch) + ',' + str(running_loss))
                    running_loss = 0.0
                if n % 1000 == 0:
                    if best_accuracy < 100 * correct / total:
                        best_accuracy = 100 * correct / total
                        save_checkpoint(state=epoch,
                                        filename='./ConvNeuralNetwork/savedCNN/dropouts/DropoutBestAcc.pth')
                    else:
                        print("accuracy did not improve, no new file saved")

            # if self.demo_mode:
            #    log_loss(str(epoch) + ',' + str(epoch_loss))

            train_loss.append(epoch_loss)
            train_accuracy.append(100 * correct / total)

        write_training_performance(train_loss, train_accuracy, label_list, predicted_list, self.features)
        print('Finished Training')

def save_checkpoint(state, filename):
    """Save checkpoint if a new best is achieved"""
    print ("=> Saving a new best to dropouts/DropoutBestAcc.pth")
    torch.save(state, filename)  # save checkpoint


def log_loss(log_text):
    f = open('logloss.txt', 'a')
    f.write(log_text + '\n')
    f.close()
