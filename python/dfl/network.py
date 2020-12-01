import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class unet:
    a = 0

class FeatureNet(nn.Module):
    def getWidth(self, n):
        return int(32 * 2 ** ((n-2)/5))
    def __init__(self, n_tasks, n_classes, error_type):
        super(FeatureNet, self).__init__()
        self.n_tasks = n_tasks
        self.n_classes = n_classes
        self.error_type = error_type
        self.batnorm1 = nn.BatchNorm1d(1)
        self.batnorm2 = nn.BatchNorm1d(self.getWidth(2))
        self.batnorm3 = nn.BatchNorm1d(self.getWidth(3))
        self.batnorm4 = nn.BatchNorm1d(self.getWidth(4))
        self.batnorm5 = nn.BatchNorm1d(self.getWidth(5))
        self.batnorm6 = nn.BatchNorm1d(self.getWidth(6))
        self.batnorm7 = nn.BatchNorm1d(self.getWidth(7))
        self.batnorm8 = nn.BatchNorm1d(self.getWidth(8))
        self.batnorm9 = nn.BatchNorm1d(self.getWidth(9))
        self.batnorm10 = nn.BatchNorm1d(self.getWidth(10))
        self.batnorm11 = nn.BatchNorm1d(self.getWidth(11))
        self.batnorm12 = nn.BatchNorm1d(self.getWidth(12))
        self.batnorm13 = nn.BatchNorm1d(self.getWidth(13))
        self.batnorm14 = nn.BatchNorm1d(self.getWidth(14))
        self.batnorm15 = nn.BatchNorm1d(self.getWidth(15))
        self.pad = 1
        self.conv1 = nn.Conv1d(1, 1, 3, 2, padding=self.pad)
        self.conv2 = nn.Conv1d(1, self.getWidth(2), 3, 1, padding=self.pad)
        self.conv3 = nn.Conv1d(self.getWidth(2), self.getWidth(3), 3, 2, padding=self.pad)
        self.conv4 = nn.Conv1d(self.getWidth(3), self.getWidth(4), 3, 2, padding=self.pad)
        self.conv5 = nn.Conv1d(self.getWidth(4), self.getWidth(5), 3, 2, padding=self.pad)
        self.conv6 = nn.Conv1d(self.getWidth(5), self.getWidth(6), 3, 2, padding=self.pad)
        self.conv7 = nn.Conv1d(self.getWidth(6), self.getWidth(7), 3, 2, padding=self.pad)
        self.conv8 = nn.Conv1d(self.getWidth(7), self.getWidth(8), 3, 2, padding=self.pad)
        self.conv9 = nn.Conv1d(self.getWidth(8), self.getWidth(9), 3, 2, padding=self.pad)
        self.conv10 = nn.Conv1d(self.getWidth(9), self.getWidth(10), 3, 2, padding=self.pad)
        self.conv11 = nn.Conv1d(self.getWidth(10), self.getWidth(11), 3, 2, padding=self.pad)
        self.conv12 = nn.Conv1d(self.getWidth(11), self.getWidth(12), 3, 2, padding=self.pad)
        self.conv13 = nn.Conv1d(self.getWidth(12), self.getWidth(13), 3, 2, padding=self.pad)
        self.conv14 = nn.Conv1d(self.getWidth(13), self.getWidth(14), 3, 2, padding=self.pad)
        self.conv15 = nn.Conv1d(self.getWidth(14), self.getWidth(15), 3)
        self.finalconv = nn.ModuleList()
        for id in range(self.n_tasks):
            self.finalconv.append(nn.Conv1d(self.getWidth(15), n_classes[id], 1))


    def forward(self, x, task_id):
        x = F.leaky_relu(self.batnorm1(self.conv1(x)))
        # x = F.dropout(x, 0.5)
        x = F.leaky_relu(self.batnorm2(self.conv2(x)))
        x = F.leaky_relu(self.batnorm3(self.conv3(x)))
        x = F.leaky_relu(self.batnorm4(self.conv4(x)))
        x = F.leaky_relu(self.batnorm5(self.conv5(x)))
        x = F.leaky_relu(self.batnorm6(self.conv6(x)))
        x = F.leaky_relu(self.batnorm7(self.conv7(x)))
        x = F.leaky_relu(self.batnorm8(self.conv8(x)))
        x = F.leaky_relu(self.batnorm9(self.conv9(x)))
        x = F.leaky_relu(self.batnorm10(self.conv10(x)))
        x = F.leaky_relu(self.batnorm11(self.conv11(x)))
        x = F.leaky_relu(self.batnorm12(self.conv12(x)))
        x = F.leaky_relu(self.batnorm13(self.conv13(x)))
        x = F.leaky_relu(self.batnorm14(self.conv14(x)))
        x = F.leaky_relu(self.batnorm15(self.conv15(x)))
        x = F.avg_pool1d(x, 2)
        x = x.mean(2, keepdim=True)
        # # LINEAR LAYER
        x = self.finalconv[task_id](x)
        x = x.reshape([-1, self.n_classes[task_id]])
        if self.error_type[task_id] == 1:
            x = F.softmax(x, dim=1)
        else:
            x = torch.sigmoid(x)
        return x


if __name__ =='__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.backends.cudnn.benchmark = True

    n_tasks = 2
    n_classes = []
    label1 = ['residential_area', 'metro_station', 'city_center', 'home', 'office', 'park', 'forest_path', 'train', 'bus', 'library', 'beach', 'cafe/restaurant', 'grocery_store', 'tram', 'car']
    label2 = ['v', 'm', 'b', 'c', 'p', 'o', 'f']
    n_classes.append(len(label1))
    n_classes.append(len(label2))
    error_type = []
    error_type.append(1)
    error_type.append(2)


    net = FeatureNet(n_tasks, n_classes, error_type)
    net.to(device)
    # print(net)
    a = torch.randn(5,1,2**15)
    # print(a)
    a = a.to(device)
    print(net.forward(a, 0)[0])
    print(net.forward(a, 0).shape)
    print(net.forward(a, 1)[0])
    print(net.forward(a, 1).shape)
    # print(net.forward(a)[1].shape)