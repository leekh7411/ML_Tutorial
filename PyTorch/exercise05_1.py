import torch
from torch.autograd import Variable

x_data = Variable(torch.Tensor([[1.0],[2.0],[3.0]]))
y_data = Variable(torch.Tensor([[4.0],[5.0],[6.0]]))


# STEP 1 : Create Model using Variable!

class Model(torch.nn.Module):
    # set subclass torch.nn.Module
    def __init__(self):
        super(Model,self).__init__()# Subclass initialize first
        self.linear = torch.nn.Linear(1,1)# Pytorch provide Linear API and parameter (1,1) means that input data and output data each will be only one

    def forward(self,x):
        # forward accept input data. And return the output data.
        y_pred = self.linear(x)
        return y_pred

# This is our network
model = Model()


# STEP 2 : Construct loss and optimizer!

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01) # Stochastic Gradient Descent


# STEP 3 : Training! forward, loss, backward, step

# training loop
epoch_size = 500
for epoch in range(epoch_size):
    # Forward Pass!
    y_pred = model(x_data)

    # Compute and print loss
    loss = criterion(y_pred,y_data)
    print(epoch,loss.data[0])

    # zero gradients, perform a backward pass, and update the weights!
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # After training....
    hour_var = Variable(torch.Tensor([[4.0]]))
    print("predict (after training)", 4, model.forward(hour_var).data[0][0])


