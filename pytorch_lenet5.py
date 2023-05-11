"""PyTorch implementation of the Lenet5 model."""

import torch
from torch.nn import Module
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms

import datetime
import time

class PyTorchLenet5Mod(Module):
    def __init__(self, initializer, dropout):
        super(PyTorchLenet5Mod, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)#, bias=True)
        initializer(self.conv1.weight)
        self.conv2 = nn.Conv2d(6, 16, 5)#, bias=True)
        initializer(self.conv2.weight)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(400, 120)#, bias=True)
        initializer(self.fc1.weight)
        self.fc2 = nn.Linear(120, 84)#, bias=True)
        initializer(self.fc2.weight)
        self.fc3 = nn.Linear(84, 10)#, bias=True)
        initializer(self.fc3.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, is_training=False):
        # pool size = 2
        # input size = (28, 28), output size = (14, 14), output channel = 6
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        # pool size = 2
        # input size = (10, 10), output size = (5, 5), output channel = 16
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        # flatten as one dimension
        # x = x.view(x.size()[0], -1)
        x = x.view(-1, 16*5*5)

        x = self.fc1(x)
        if is_training:
            x = self.dropout(x)

        # input dim = 16*5*5, output dim = 120
        x = torch.nn.functional.relu(x)

        x = self.fc2(x)

        if is_training:
            x = self.dropout(x)

        # input dim = 120, output dim = 84
        x = torch.nn.functional.relu(x)
        # input dim = 84, output dim = 10
        x = self.fc3(x)
        return x

    def train_pytorch(self, optimizer, epoch, train_loader, device, data_type, log_interval):
        # State that you are training the model
        self.train()

        # define loss function
        loss_fn = torch.nn.CrossEntropyLoss()

        if data_type == 'mixed':
            scaler = torch.cuda.amp.GradScaler()

        # Iterate over batches of data
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # if device == 'gpu':
            #     data, target = data.cuda(), target.cuda()

            # Wrap the input and target output in the `Variable` wrapper
            # data, target = Variable(data), Variable(target)

            # Clear the gradients, since PyTorch accumulates them
            optimizer.zero_grad()

            if data_type == 'mixed':
                with torch.cuda.amp.autocast():
                    # Forward propagation
                    output = self(data, is_training=True)

                    loss = loss_fn(output, target)
                    # print(loss.device)

                # Backward propagation
                scaler.scale(loss).backward()

                # Update the parameters(weight,bias)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward propagation
                output = self(data, is_training=True)

                loss = loss_fn(output, target)
                # print(loss.device)

                # Backward propagation
                loss.backward()

                # Update the parameters(weight,bias)
                optimizer.step()

            if log_interval == -1:
                continue

            # print log
            if batch_idx % log_interval == 0:
                print('Train set, Epoch {}\tLoss: {:.6f}'.format(
                    epoch, loss.item()))
        # return model

    def test_pytorch(self, test_loader, device, data_type):
        # State that you are testing the model; this prevents layers e.g. Dropout to take effect
        self.eval()

        with torch.no_grad():

            # Init loss & correct prediction accumulators
            test_loss = 0
            correct = 0

            # define loss function
            loss_fn = torch.nn.CrossEntropyLoss(size_average=False)
            # Iterate over data
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                # data, target = Variable(data), Variable(target)
                if data_type == 'mixed':
                    with torch.cuda.amp.autocast():
                        # Forward propagation
                        output = self(data).detach()

                        # Calculate & accumulate loss
                        test_loss += loss_fn(output, target).detach()

                        # Get the index of the max log-probability (the predicted output label)
                        # pred = np.argmax(output.data, axis=1)
                        pred = output.data.argmax(dim=1)

                        # If correct, increment correct prediction accumulator
                        correct = correct + (pred == target.data).sum()
                else:
                    # Forward propagation
                    output = self(data).detach()

                    # Calculate & accumulate loss
                    test_loss += loss_fn(output, target).detach()

                    # Get the index of the max log-probability (the predicted output label)
                    # pred = np.argmax(output.data, axis=1)
                    pred = output.data.argmax(dim=1)

                    # If correct, increment correct prediction accumulator
                    correct = correct + (pred == target.data).sum()

            # Print log
            test_loss /= len(test_loader.dataset)
            print('\nTest set, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

            return 100. * correct / len(test_loader.dataset)
        
        
def generate_pytorch_dataloader(X_train_padded, X_test_padded, X_test_padded_ext, y_train, y_test, y_test_ext, batch_size, device):

    X_torch_train_mnist = torch.from_numpy(X_train_padded).view(X_train_padded.shape[0], 1, 32, 32).type(torch.float32).to(device)
    X_torch_test_mnist = torch.from_numpy(X_test_padded).view(X_test_padded.shape[0], 1, 32, 32).type(torch.float32).to(device)
    X_torch_test_mnist_ext = torch.from_numpy(X_test_padded_ext).view(X_test_padded_ext.shape[0], 1, 32, 32).type(torch.float32).to(device)
            
    y_torch_train_mnist = torch.LongTensor(y_train).to(device)
    y_torch_test_mnist = torch.LongTensor(y_test).to(device)
    y_torch_test_mnist_ext = torch.LongTensor(y_test_ext).to(device)
    

    pytorch_lenet5_train_dataset = TensorDataset(X_torch_train_mnist, y_torch_train_mnist)
    pytorch_lenet5_test_dataset = TensorDataset(X_torch_test_mnist, y_torch_test_mnist)
    pytorch_lenet5_test_dataset_ext = TensorDataset(X_torch_test_mnist_ext, y_torch_test_mnist_ext)

    pytorch_train_loader = DataLoader(pytorch_lenet5_train_dataset, batch_size=batch_size, shuffle=False)
    pytorch_test_loader = DataLoader(pytorch_lenet5_test_dataset, batch_size=batch_size, shuffle=False)
    pytorch_test_loader_ext = DataLoader(pytorch_lenet5_test_dataset_ext, batch_size=batch_size, shuffle=False)
    
#     X_torch_train = torch.from_numpy(X_train_padded).view(X_train_padded.shape[0], review_length).to(device)
#     X_torch_test = torch.from_numpy(X_test_padded).view(X_test_padded.shape[0], review_length).to(device)
#     X_torch_test_ext = torch.from_numpy(X_test_padded_ext).view(X_test_padded_ext.shape[0], review_length).to(device)

#     y_torch_train = torch.FloatTensor(y_train).to(device)
#     y_torch_test = torch.FloatTensor(y_test).to(device)
#     y_torch_test_ext = torch.FloatTensor(y_test_ext).to(device)


#     pytorch_train_dataset = TensorDataset(X_torch_train, y_torch_train)
#     pytorch_test_dataset = TensorDataset(X_torch_test, y_torch_test)
#     pytorch_test_dataset_ext = TensorDataset(X_torch_test_ext, y_torch_test_ext)

#     pytorch_train_loader = DataLoader(pytorch_train_dataset, batch_size=batch_size, shuffle=False)
#     pytorch_test_loader = DataLoader(pytorch_test_dataset, batch_size=batch_size, shuffle=False)
#     pytorch_test_loader_ext = DataLoader(pytorch_test_dataset_ext, batch_size=batch_size, shuffle=False)
    
    return pytorch_train_loader, pytorch_test_loader, pytorch_test_loader_ext


def pytorch_training_phase(model, optimizer, train_loader, test_loader, n_epochs, device, data_type, experiment):
    train_start_timestamp = datetime.datetime.now()
    start = time.time()

    for epoch in range(1, n_epochs+1):
        model.train_pytorch(optimizer, epoch, train_loader, device, data_type, log_interval=60000)

    training_time = time.time() - start
    train_end_timestamp = datetime.datetime.now()

    
    start = time.time()
    accuracy = model.test_pytorch(test_loader, device, data_type)
    inference_time = (time.time() - start)

    #Save the model
    torch.save(model.state_dict(), './models/lenet5/{}/model'.format(experiment))

    return training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp

def pytorch_inference_phase(model, experiment, pytorch_test_loader_ext, device, data_type):
    model.load_state_dict(torch.load('./models/lenet5/{}/model'.format(experiment)))
    model.eval()

    inference_start_timestamp = datetime.datetime.now()
    accuracy = model.test_pytorch(pytorch_test_loader_ext, device, data_type)
    inference_end_timestamp = datetime.datetime.now()
    print('Accuracy: {}'.format(accuracy))
    
    return inference_start_timestamp, inference_end_timestamp


def PyTorchLenet5(experiment, X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist, batch_size, n_epochs, learning_rate, data_type, device, initializer):
    if data_type == 'mixed':
        X_torch_train_mnist = torch.from_numpy(training_mnist_norm).view(training_mnist_norm.shape[0], 1, 32, 32).type(torch.float16).to(device)
        X_torch_test_mnist = torch.from_numpy(testing_mnist_norm).view(testing_mnist_norm.shape[0], 1, 32, 32).type(torch.float16).to(device)
    else:
        X_torch_train_mnist = torch.from_numpy(training_mnist_norm).view(training_mnist_norm.shape[0], 1, 32, 32).type(torch.float32).to(device)
        X_torch_test_mnist = torch.from_numpy(testing_mnist_norm).view(testing_mnist_norm.shape[0], 1, 32, 32).type(torch.float32).to(device)

    y_torch_train_mnist = torch.LongTensor(y_train_mnist).to(device)
    y_torch_test_mnist = torch.LongTensor(y_test_mnist).to(device)

    pytorch_lenet5_train_dataset = TensorDataset(X_torch_train_mnist, y_torch_train_mnist)
    pytorch_lenet5_test_dataset = TensorDataset(X_torch_test_mnist, y_torch_test_mnist)

    pytorch_lenet5_train_loader = DataLoader(pytorch_lenet5_train_dataset, batch_size=batch_size, shuffle=False)
    pytorch_lenet5_test_loader = DataLoader(pytorch_lenet5_test_dataset, batch_size=batch_size, shuffle=False)

    pytorchLenet5 = PyTorchLenet5Mod(initializer)

    pytorchLenet5 = pytorchLenet5.to(device)

    optimizer = SGD(pytorchLenet5.parameters(), lr=learning_rate)

    train_start_timestamp = datetime.datetime.now()
    start = time.time()

    for epoch in range(1, n_epochs+1):
        train_pytorch(pytorchLenet5, optimizer, epoch, pytorch_lenet5_train_loader,
                      device, log_interval=60000)

    training_time = time.time() - start
    train_end_timestamp = datetime.datetime.now()

    inference_start_timestamp = datetime.datetime.now()
    start = time.time()
    accuracy = test_pytorch(pytorchLenet5, pytorch_lenet5_test_loader, device).item()
    inference_time = (time.time() - start) / testing_mnist_norm.shape[0]
    inference_end_timestamp = datetime.datetime.now()

    # torch.save(pytorchLenet5.state_dict(), './models/{}'.format(experiment))

    return training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp 

#training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp, inference_start_timestamp, inference_end_timestamp = PyTorchLenet5(experiment, training_mnist_norm, y_train_mnist, testing_mnist_norm, y_test_mnist, batch_size, n_epochs, learning_rate, data_type, device_dict[device][framework], weight_initialization_dict[weight_initialization][framework])
