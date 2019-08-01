import torch
import torch.nn as nn
import torch.nn.functional as F
# from RND import PredictNet, RandomNet

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

class RandomNet(nn.Module):
    def __init__(self):
        super().__init__()

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(84)
        convh = conv2d_size_out(84)
        linear_layer_input = 56448

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(linear_layer_input, 512)


    def forward(self, x):
        out = F.selu(self.bn1(self.conv1(x)))
        out = self.fc(out.reshape((out.shape[0], -1)))

        return out

class PredictNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.head = nn.Sequential(
        #     nn.Linear(3, 3*10),
        #     nn.SELU()
        # )
        
        linear_in = 28224

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(linear_in, 512)

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(84))
        convh = conv2d_size_out(conv2d_size_out(84))
        linear_layer_input = convw * convh * 32

    def forward(self, x):
        # out = self.head(x)
        out = F.selu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = F.selu(self.bn2(self.conv2(out)))
        # print(out.shape)
        out = self.fc(out.reshape((out.shape[0], -1)))
        return out






class DuelingDQN(nn.Module):
    def __init__(self, action_space):
        super().__init__()

        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(84)))
        linear_layer_input = convw * convh * 32
        val_linear_input = 1568
        adv_linear_input = 352

        # self.head = nn.Sequential(
        #     nn.Linear(3, 3*10),
        #     nn.SELU()
        # )

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
      
      
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.val = nn.Linear(3872, 1)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 32, kernel_size=5, stride=2, padding=2)
        self.bn5 = nn.BatchNorm2d(32)
        self.adv = nn.Linear(3872, action_space)


        

        # self.adv = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(64),
        #     nn.SELU(),
        #     nn.Conv2d(64, 32, kernel_size=5, stride=2),
        #     nn.BatchNorm2d(32),
        #     nn.SELU(),
        #     nn.Linear(linear_layer_input, action_space)
        # )


    def forward(self, x):
        # out = self.head(x)
        out = F.selu(self.bn1(self.conv1(x)))
        val_out = F.selu(self.bn2(self.conv2(out)))
        val_out = F.selu(self.bn3(self.conv3(val_out)))
        val_out = self.val(val_out.reshape(out.shape[0], -1)).reshape(out.shape[0], 1)



        adv_out = F.selu(self.bn4(self.conv4(out)))
        adv_out = F.selu(self.bn5(self.conv5(adv_out)))
        adv_out = self.adv(adv_out.reshape(out.shape[0], -1))


        adv_mean = adv_out.mean(dim=1, keepdim=True)
        q = val_out + adv_out - adv_mean

        return q


# device = torch.device('cpu')
# pred_net = PredictNet().to(device)
# rand_net = RandomNet().to(device)
# net = DuelingDQN(54).to(device)
# x = torch.rand((256,3,84,84))
# print(net(x).shape)
