import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader

class DiabetesDataLoader(Dataset):
    def __init__(self):
        xy = np.loadtxt('./data/diabetes.csv.gz',delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = (torch.from_numpy(xy[:, 0:-1]))
        self.y_data = (torch.from_numpy(xy[:, [-1]]))

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.len


dataset = DiabetesDataLoader()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2) # workers means Multi Processing




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


for epoch in range(2):
    for i,data in enumerate(train_loader,0):
        # get the input
        inputs,labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs),Variable(labels)

        # Forward Pass!
        y_pred = model(inputs)

        # Compute and print Loss
        loss = criterion(y_pred,labels)
        print(epoch,i,loss.data[0])

        # zero gradient and backward and updates the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


