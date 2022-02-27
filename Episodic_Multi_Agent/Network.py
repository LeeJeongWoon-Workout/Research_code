import torch
import torch.nn.functional as F
import torch.nn as nn


class MuNet(nn.Module):
    def __init__(self):
        super(MuNet,self).__init__()
        self.fc1=nn.Linear(3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,1)


    def forward(self,x):

        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        mu=torch.tanh(self.fc3(x))*2

        return mu


class QNet(nn.Module):

    def __init__(self,):

        super(QNet,self).__init__()
        self.fc_s=nn.Linear(3,64)
        self.fc_a=nn.Linear(1,64)
        self.fc_q=nn.Linear(128,32)
        self.fc_out=nn.Linear(32,1)

    def forward(self,s,a):

        h1=F.relu(self.fc_s(s))
        h2=F.relu(self.fc_a(a))
        cat=torch.cat([h1,h2],dim=1)
        q=F.relu(self.fc_q(cat))

        return self.fc_out(q)