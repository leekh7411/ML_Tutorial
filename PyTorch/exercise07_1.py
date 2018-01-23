import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
xy = np.loadtxt('./data/diabetes.csv.gz',delimiter=',',dtype=np.float32)
len = xy.shape[0]
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))



class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.l1 = torch.nn.Linear(8,60)
        self.l2 = torch.nn.Linear(60,14)
        self.l3 = torch.nn.Linear(14,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred




model = Model()

criterion = torch.nn.BCEWithLogitsLoss(size_average=True)
optimizer = torch.optim.ASGD(model.parameters(),lr=0.1)



for epoch in range(1000):
    # Foward Pass
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred,y_data)
    print(epoch,loss.data[0])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test_var = x_data
print(test_var)
print('Result:', model.forward(test_var))
