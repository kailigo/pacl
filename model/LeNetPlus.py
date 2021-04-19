import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as breakpoint




class LeNetPlus(nn.Module):
    def __init__(self):
        super(LeNetPlus, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)

        self.fc1 = nn.Linear(6272, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):            
        
        x = F.instance_norm(x)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.max_pool2d(x, kernel_size=2)

        # breakpoint()        
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.max_pool2d(x, kernel_size=2, )
        x = F.dropout(x)

        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x)
        
        # breakpoint()
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        # x = F.relu(x)
        # x = F.leaky_relu(x, 0.2)

        prop = self.fc2(x)
        
        return prop, x



class LeNetPlus_Triplet(nn.Module):
    def __init__(self):
        super(LeNetPlus_Triplet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=2)        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2)

        self.fc1 = nn.Linear(6272, 512)
        # self.fc2 = nn.Linear(512, 10)

    def forward(self, x):   


        x = F.instance_norm(x)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.max_pool2d(x, kernel_size=2)

        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.max_pool2d(x, kernel_size=2, )
        x = F.dropout(x)

        x = F.leaky_relu(self.conv5(x), 0.2)
        x = F.leaky_relu(self.conv6(x), 0.2)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = F.relu(x)
        # prop = self.fc2(x)
        
        return x



class Predictor(nn.Module):
    def __init__(self, inc=512, num_class=10):
        super(Predictor, self).__init__()
        self.fc = nn.Linear(inc, num_class)
        
    def forward(self, x):
        x_out = self.fc(x)
        return x_out



class Instance_Classifier(nn.Module):
    def __init__(self, inc, hidden_dim):
        super(Instance_Classifier, self).__init__()
        self.dc_ip1 = nn.Linear(inc, hidden_dim)
        self.dc_relu1 = nn.ReLU()
        self.dc_drop1 = nn.Dropout(p=0.5)
        self.clssifer = nn.Linear(hidden_dim, 2)
        
    def forward(self, x):
        x=self.dc_drop1(self.dc_relu1(self.dc_ip1(x)))
        x=self.clssifer(x)
        return x
