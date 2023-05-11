import torch
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt
import random
import time
import csv
# import GPUtil
# import psutil
import datetime

import torch
from torch.nn import Module
from torch import nn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model

from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms
import sys

device_dict = {
    'cpu': {
        'PyTorch': 'cpu',
        'Keras': '/cpu:0',
        'TensorFlow': '/CPU:0'
    },
    'gpu': {
        'PyTorch': 'cuda',
        'Keras': '/gpu:0',
        'TensorFlow': '/GPU:0'
    }
}

weight_initialization_dict = { 
    'xavier': {
        'PyTorch': torch.nn.init.xavier_normal_,
        'Keras': tf.keras.initializers.GlorotNormal,
        'TensorFlow': tf.compat.v1.initializers.glorot_normal
    },
    'ones': {
        'PyTorch': torch.nn.init.ones_,
        'Keras': tf.keras.initializers.Ones,
        'TensorFlow': tf.compat.v1.initializers.ones
    },
    'he': {
        'PyTorch': torch.nn.init.kaiming_normal_,
        'Keras': tf.keras.initializers.HeNormal,
        'TensorFlow': tf.compat.v1.keras.initializers.he_normal
    }
}

i = 0
training_size = 1
batch_size = 256
n_epochs = 5
learning_rate = 0.01
data_type = 'float32'
device = sys.argv[2]
weight_initialization = 'xavier'
framework = sys.argv[1]
dropout = 0

import torch
from torch.nn import Module
from torch import nn
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib as plt
import random
import time
import csv
# import GPUtil
# import psutil
import tensorflow as tf
import datetime

collect_time = 0

#Making sure the tensorflow doesn't take up all the VRAM available on GPU
if device == 'gpu':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# print('GPU mem: {}'.format((GPUtil.getGPUs()[0].memoryUsed / GPUtil.getGPUs()[0].memoryTotal) * 100))
# print('CPU mem: {}'.format(psutil.virtual_memory().used / psutil.virtual_memory().total))

training_size_list = [0.1, 0.2, 0.3, 0.4, 0.5]

batch_size_list = [32, 64, 128, 256, 512, 1024]

n_epochs_list = [10, 20, 30]

learning_rate_list = [0.1, 0.01, 0.001]

data_type_list = ['float32', 'mixed']

device_dict = {
    'gpu': {
        'PyTorch': 'cuda',
        'Keras': '/gpu:0',
        'TensorFlow': '/GPU:0'
    },
    'cpu': {
        'PyTorch': 'cpu',
        'Keras': '/cpu:0',
        'TensorFlow': '/CPU:0'
    }
}

weight_initialization_dict = {
    'xavier': {
        'PyTorch': torch.nn.init.xavier_normal_,
        'Keras': tf.keras.initializers.GlorotNormal,
        'TensorFlow': tf.compat.v1.initializers.glorot_normal
    },
    'ones': {
    'PyTorch': torch.nn.init.ones_,
    'Keras': tf.keras.initializers.Ones,
    'TensorFlow': tf.compat.v1.initializers.ones
    },
    'he': {
        'PyTorch': torch.nn.init.kaiming_normal_,
        'Keras': tf.keras.initializers.HeNormal,
        'TensorFlow': tf.compat.v1.keras.initializers.he_normal
    }
}

experiment = 'lenet5_{}{}_{}ts_{}batch_{}epochs_{}lr_{}dtype_{}_{}wi_{}dp'.format(framework, i,
                                                                              training_size, batch_size,
                                                                              n_epochs, learning_rate, data_type,
                                                                              device, weight_initialization, dropout)

from FetchData import fetchMNIST
X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist = fetchMNIST()

#we try to minimize the randomness as much as possible
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
# tf.random.set_seed(0)
random.seed(0)
np.random.seed(0)


print('Training {}'.format(experiment))

training_time = 0
inference_time = 0
accuracy = 0

train_start_timestamp = 0
train_end_timestamp = 0

X_train_mnist_temp = X_train_mnist[:int(training_size*len(X_train_mnist))]
y_train_mnist = y_train_mnist[:len(X_train_mnist_temp)]

training_mnist = X_train_mnist_temp.reshape(X_train_mnist_temp.shape[0], 28, 28, 1)
testing_mnist = X_test_mnist.reshape(X_test_mnist.shape[0], 28, 28, 1)

training_mnist = np.pad(training_mnist, ((0,0),(2,2),(2,2), (0,0)), 'constant')
testing_mnist = np.pad(testing_mnist, ((0,0),(2,2),(2,2), (0,0)), 'constant')

if data_type == 'mixed':
    training_mnist_norm = np.array(training_mnist / 255.0, dtype=np.float16)
    testing_mnist_norm = np.array(testing_mnist / 255.0, dtype=np.float16)
else:
    training_mnist_norm = np.array(training_mnist / 255.0)
    testing_mnist_norm = np.array(testing_mnist / 255.0)

if framework == 'PyTorch':
    from torch.utils.data import Dataset, DataLoader
    from torch.nn import CrossEntropyLoss
    from torch.optim import SGD
    from torch.utils.data import DataLoader, TensorDataset
    from torchvision.transforms import ToTensor
    from torch.autograd import Variable
    from torchvision import transforms

    class PyTorchLenet5Mod(Module):
        def __init__(self, initializer, data_type):
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

    def train_pytorch(model, optimizer, epoch, train_loader, device, log_interval):
        # State that you are training the model
        model.train()

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
                    output = model(data, is_training=True)

                    loss = loss_fn(output, target)
                    # print(loss.device)

                # Backward propagation
                scaler.scale(loss).backward()

                # Update the parameters(weight,bias)
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward propagation
                output = model(data, is_training=True)

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

    def test_pytorch(model, test_loader, device):
        # State that you are testing the model; this prevents layers e.g. Dropout to take effect
        model.eval()

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
                        output = model(data).detach()

                        # Calculate & accumulate loss
                        test_loss += loss_fn(output, target).detach()

                        # Get the index of the max log-probability (the predicted output label)
                        # pred = np.argmax(output.data, axis=1)
                        pred = output.data.argmax(dim=1)

                        # If correct, increment correct prediction accumulator
                        correct = correct + (pred == target.data).sum()
                else:
                    # Forward propagation
                    output = model(data).detach()

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

        pytorchLenet5 = PyTorchLenet5Mod(initializer, data_type)

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

    training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp = PyTorchLenet5(experiment, training_mnist_norm, y_train_mnist, testing_mnist_norm, y_test_mnist, batch_size, n_epochs, learning_rate, data_type, device_dict[device][framework], weight_initialization_dict[weight_initialization][framework])


    print('{}\tTraining time: {}\tInference time: {}\tAccuracy: {}'.format(experiment, training_time,
                                                                           inference_time, accuracy))


if framework == 'Keras':

    import os
    os.environ['TF2_BEHAVIOR'] = '1'
    import tensorflow as tf

    tf.random.set_seed(0)

    if data_type == 'mixed':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    
    keras_lenet5_model = keras.models.Sequential([
        keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='relu', kernel_initializer=weight_initialization_dict[weight_initialization][framework], input_shape=(32,32,1)),# dtype=data_type_dict[data_type][framework]), #C1
        keras.layers.MaxPool2D(strides=2), #S2
        keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='relu', kernel_initializer=weight_initialization_dict[weight_initialization][framework]),# dtype=data_type_dict[data_type][framework]), #C3
        keras.layers.MaxPool2D(strides=2), #S4
        keras.layers.Flatten(), #Flatten
        # keras.layers.Dropout(dropout),
        keras.layers.Dense(120, activation='relu', kernel_initializer=weight_initialization_dict[weight_initialization][framework]),# dtype=data_type_dict[data_type][framework]), #C5
        keras.layers.Dropout(dropout),
        keras.layers.Dense(84, activation='relu', kernel_initializer=weight_initialization_dict[weight_initialization][framework]),# dtype=data_type_dict[data_type][framework]), #F6
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation='softmax', kernel_initializer=weight_initialization_dict[weight_initialization][framework]),# dtype=data_type_dict[data_type][framework]) #Output layer
    ])


    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    
    if data_type == 'mixed':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    keras_lenet5_model.compile(optimizer=optimizer,
                               loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

    start = time.time()

    train_start_timestamp = datetime.datetime.now()
    with tf.device(device_dict[device][framework]):
        keras_lenet5_model.fit(training_mnist_norm,
                             y_train_mnist,
                             batch_size=batch_size,
                             epochs=n_epochs,
                             shuffle=False)#,
                             #callbacks=[callback])
    train_end_timestamp = datetime.datetime.now()

    training_time = time.time() - start

    inference_start_timestamp = datetime.datetime.now()
    start = time.time()
    accuracy = keras_lenet5_model.evaluate(testing_mnist_norm, y_test_mnist, batch_size=batch_size)[1]
    inference_time = (time.time() - start) / testing_mnist_norm.shape[0]
    inference_end_timestamp = datetime.datetime.now()


    # if device == 'gpu':
    #     training_time, inference_time, accuracy, cpu_utilization, cpu_mem, gpu_utilization, gpu_mem = KerasLenet5(experiment, training_mnist_norm, y_train_mnist, testing_mnist_norm, y_test_mnist, batch_size, n_epochs, learning_rate, data_type, device_dict[device][framework], weight_initialization_dict[weight_initialization][framework])
    # else:
    #     training_time, inference_time, accuracy, cpu_utilization, cpu_mem, _, _ = KerasLenet5(experiment, training_mnist_norm, y_train_mnist, testing_mnist_norm, y_test_mnist, batch_size, n_epochs, learning_rate, data_type, device_dict[device][framework], weight_initialization_dict[weight_initialization][framework])

    # keras_lenet5_model.save('./models/{}'.format(experiment))

    print('{}\tTraining time: {}\tInference time: {}\tAccuracy: {}'.format(experiment, training_time,
                                                                          inference_time, accuracy))

if framework == 'TensorFlow':
    # the first version of tensorflow needs to be used, as tensorflow 2.0 uses keras by default
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    # from tensorflow.compat.v1.layers import flatten


    tf.compat.v1.set_random_seed(0)

    import os
    os.environ['TF2_BEHAVIOR'] = '1'
    import tensorflow as tf

    def TensorFlowLenet5(experiment, training_mnist_norm, y_train_mnist, testing_mnist_norm, y_test_mnist, batch_size, n_epochs, learning_rate, data_type, device, initializer):
        with tf.device(device):
            with tf.compat.v1.variable_scope(name_or_scope='TensorFlowLenet5', reuse=False, initializer=initializer):#, dtype=data_type_dict[data_type][framework]):

                is_training = False
                
                x = tf.compat.v1.placeholder(tf.float32, (None, 32, 32, 1))
                # print(x)
                y = tf.compat.v1.placeholder(tf.uint8, (None))
                one_hot_y = tf.one_hot(y, 10)
                # with tf.compat.v1.variable_scope("TensorFlowLenet5", reuse=tf.compat.v1.AUTO_REUSE):
                # print(data_type)
                #Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
                conv1_weight=tf.compat.v1.get_variable("Wconv1_{}".format(experiment), shape=[5, 5, 1, 6])#, mean=mu, stddev=sigma))
                # conv1_weight = tf.Session().run(tf.cast(conv1_weight, data_type))
                # print(conv1_weight)
                # conv1_bias=tf.Variable(tf.zeros(6))
                #conv1=tf.add(tf.matmul(x, conv1_weight), conv1_bias)
                # conv1=tf.compat.v1.nn.conv2d(x, tf.compat.v1.cast(conv1_weight, data_type), strides=[1, 1, 1, 1], padding='VALID')#+conv1_bias
                conv1=tf.compat.v1.nn.conv2d(x, conv1_weight, strides=[1, 1, 1, 1], padding='VALID')

                #Activation.
                conv1=tf.nn.relu(conv1)

                #Pooling. Input = 28x28x6. Output = 14x14x6.
                conv1=tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                #Layer 2: Convolutional. Output = 10x10x16.
                conv2_weight=tf.compat.v1.get_variable("Wconv2_{}".format(experiment), shape=[5, 5, 6, 16])#tf.Variable(initializer(dtype=data_type)(shape=(5, 5, 6, 16)))#, mean=mu, stddev=sigma))
                #conv2_bias=tf.Variable(tf.zeros(16))
                conv2=tf.compat.v1.nn.conv2d(conv1, conv2_weight, strides=[1, 1, 1, 1], padding='VALID')#+conv2_bias

                #Activation.
                conv2=tf.nn.relu(conv2)

                #Pooling. Input = 10x10x16. Output = 5x5x16.
                conv2=tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

                #Flatten. Input = 5x5x16. Output = 400.
                shape=conv2.get_shape().as_list()
                flatten_size=np.prod(shape[1:])
                flatten_layer=tf.reshape(conv2, (-1, flatten_size))
                #Using build in function directly : flatten_layer=flatten(conv2)

                #Layer 3: Fully Connected. Input = 400. Output = 120.
                fc1_weight=tf.compat.v1.get_variable("Wfc1_{}".format(experiment), shape=[400, 120])#tf.Variable(initializer(dtype=data_type)(shape=(400, 120)))#, mean=mu, stddev=sigma))
                #fc1_bias=tf.Variable(tf.zeros(120))
                #fc1=tf.add(tf.matmul(flatten_layer, fc1_weight), fc1_bias)
                # print(flatten_layer)
                # print(fc1_weight)
                fc1=tf.matmul(flatten_layer, fc1_weight)

                if is_training:
                    fc1 = tf.compat.v1.layers.dropout(fc1, 1-dropout)
                
                #Activation.
                fc1=tf.nn.relu(fc1)
                
                #Layer 4: Fully Connected. Input = 120. Output = 84.
                fc2_weight=tf.compat.v1.get_variable("Wfc2_{}".format(experiment), shape=[120, 84])# fc2_weight=tf.Variable(initializer(dtype=data_type)(shape=(120, 84)))#, mean=mu, stddev=sigma))
                #fc2_bias=tf.Variable(tf.zeros(84))
                #fc2=tf.add(tf.matmul(fc1, fc2_weight), fc2_bias)
                fc2=tf.matmul(fc1, fc2_weight)

                if is_training:
                    fc2 = tf.compat.v1.layers.dropout(fc2, 1-dropout)
                
                #Activation.
                fc2=tf.nn.relu(fc2)

                #Layer 5: Fully Connected. Input = 84. Output = 10.
                fc3_weight=tf.compat.v1.get_variable("Wfc3_{}".format(experiment), shape=[84, 10])#tf.Variable(initializer(dtype=data_type)(shape=(84, 10)))#, mean=mu, stddev=sigma))
                # fc3_bias=tf.Variable(tf.zeros(10))
                # logits=tf.add(tf.matmul(fc2, fc3_weight), fc3_bias)
                logits=tf.matmul(fc2, fc3_weight)

                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
                loss_operation = tf.reduce_mean(cross_entropy)
                
                optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
                if data_type == 'mixed':
                    optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
                
                training_operation = optimizer.minimize(loss_operation)

                # X = tf.placeholder(data_type_dict[data_type]['data'], [None, 32, 32, 1])
                # y = tf.placeholder(tf.int64, [None])

                config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
                if data_type == 'mixed':
                    config.graph_options.rewrite_options.auto_mixed_precision = 1
                
                with tf.compat.v1.Session(config=config) as sess:
                    sess.run(tf.compat.v1.global_variables_initializer())
                    num_examples = len(training_mnist_norm)

                    
                    train_start_timestamp = datetime.datetime.now()
                    start = time.time()
                    # print("Training...")
                    # print()
                    is_training = True
                    for i in range(1, n_epochs+1):
                        # X_train, y_train = shuffle(X_train, y_train)
                        for offset in range(0, num_examples, batch_size):
                            end = offset + batch_size
                            batch_x, batch_y = training_mnist_norm[offset:end], y_train_mnist[offset:end]
                            # print(batch_x.shape)
                            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

                        print("EPOCH {}".format(i))

                    training_time = time.time() - start
                    train_end_timestamp = datetime.datetime.now()
                    is_training = False
                    
                    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
                    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    num_examples = len(testing_mnist_norm)
                    total_accuracy = 0
                    sess = tf.compat.v1.get_default_session()
                    
                    inference_start_timestamp = datetime.datetime.now()
                    start = time.time()
                    for offset in range(0, num_examples, batch_size):
                        batch_x, batch_y = testing_mnist_norm[offset:offset+batch_size], y_test_mnist[offset:offset+batch_size]
                        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
                        total_accuracy += (accuracy * len(batch_x))
                    inference_time = (time.time() - start) / testing_mnist_norm.shape[0]
                    inference_end_timestamp = datetime.datetime.now()
                    accuracy = total_accuracy / num_examples

                    return training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp, inference_start_timestamp, inference_end_timestamp


    
    training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp, inference_start_timestamp, inference_end_timestamp = TensorFlowLenet5(experiment, training_mnist_norm, y_train_mnist, testing_mnist_norm, y_test_mnist, batch_size, n_epochs, learning_rate, data_type, device_dict[device][framework], weight_initialization_dict[weight_initialization][framework])

    print('{}\tTraining time: {}\tInference time: {}\tAccuracy: {}'.format(experiment, training_time,
                                                                           inference_time, accuracy))

results = {
    'training_time': training_time,
    'inference_time': inference_time,
    'accuracy': accuracy,
    'train_start_timestamp': train_start_timestamp,
    'train_end_timestamp': train_end_timestamp,
    'inference_start_timestamp': inference_start_timestamp,
    'inference_end_timestamp': inference_end_timestamp
}

# with open('./Results/lenet5/{}.txt'.format(experiment), 'w+', encoding='utf-8') as f:
#         for fieldName in results.keys():
#             f.write('{} = {}\n\n'.format(fieldName, results[fieldName]))

print()