import pandas as pd
import os

from PIL import Image
import torch

from torch.utils.data import DataLoader

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)
        return img, y_label

    def str(self):
        return str(self.annotations)


def load_dataset(transform, batch_size, shuffle, pin_memory, num_workers, features):
    print(os.path.abspath(os.curdir))
    dataset = MyDataset("DatasetImages",
                        "./DatasetImages/mytrain.csv", transform=transform)


    train_set, test_set = torch.utils.data.random_split(dataset, [int((len(dataset))) - int(((len(dataset)) / 4)),
                                                                  int(((len(dataset)) / 4))])
    print("total data set size: " + str(len(dataset)) + " ,training set size " + str(len(train_set)) + "test set size: " + str(len(test_set)))

    train_loader = DataLoader(dataset=train_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(dataset=test_set, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=pin_memory)

    classes = features
    return train_loader, test_loader, classes


#################################################################################
#
# Test
#
#################################################################################
if __name__ == "__main__":
    test = MyDataset()
