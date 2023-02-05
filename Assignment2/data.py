import paddle
import random

from paddle.io import Dataset
from paddle.vision.transforms import Compose, Normalize

random.seed(1217)

transform = Compose([Normalize(mean=[127.5],
                               std=[127.5],
                               data_format='CHW')])

def prepare_dataset():
    train = paddle.vision.datasets.MNIST(mode = 'train', transform=transform)
    test = paddle.vision.datasets.MNIST(mode = 'test', transform=transform)
    train_split = [(x, y) for (x, y) in train if y >= 5 or random.random() <= 0.1]
    return train, test, train_split

class MyDataset(Dataset):
    def __init__(self, xlist):
        super().__init__()
        self.x = xlist

    def __getitem__(self, idx):
        return self.x[idx]

    def __len__(self):
        return len(self.x)
