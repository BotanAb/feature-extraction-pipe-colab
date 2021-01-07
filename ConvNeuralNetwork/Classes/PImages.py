import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import os
import torchvision


class PImages:

    def __init__(self, loader):
        self.loader = loader

    def __plot(self, img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def plot(self):
        # get some random training images
        dataiter = iter(self.loader)
        images, labels = dataiter.next()

        self.__plot(torchvision.utils.make_grid(images))


#################################################################################
#
# Test
#
#################################################################################
if __name__ == "__main__":
    test = PImages()
