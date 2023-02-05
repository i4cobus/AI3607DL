import paddle
import paddle.nn.functional as F

from paddle.metric import Accuracy

class LeNet(paddle.nn.Layer):

    def __init__(self, detach_feats=False):
        super(LeNet, self).__init__()
        self.conv1 = paddle.nn.Conv2D(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = paddle.nn.MaxPool2D(kernel_size=2,  stride=2)
        self.conv2 = paddle.nn.Conv2D(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.max_pool2 = paddle.nn.MaxPool2D(kernel_size=2, stride=2)
        self.linear1 = paddle.nn.Linear(in_features=16*5*5, out_features=120)
        self.linear2 = paddle.nn.Linear(in_features=120, out_features=84)
        self.linear3 = paddle.nn.Linear(in_features=84, out_features=10)
        self.detach_feats = detach_feats

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        if self.detach_feats:
            x = x.detach()
        x = self.linear3(x)
        return x

def LeNet_model(eps=1e-8, **kwargs):
    model = paddle.Model(LeNet(**kwargs))   
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters(), epsilon=eps)

    model.prepare(
        optim,
        paddle.nn.CrossEntropyLoss(),
        Accuracy()
    )
    return model