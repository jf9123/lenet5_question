#Making sure the TensorFlow v2 is enabled
import os
os.environ['TF2_BEHAVIOR'] = '1'
import tensorflow as tf
from tensorflow import keras
import datetime
import time

tf.random.set_seed(0)

# class KerasLSTMMod(keras.models.Sequential):
    
#     def __init__(initializer, vocabulary_size, review_length,
#                  embedding_size, hidden_size, dropout):
#         #KerasLSTMMod, self
#         super().__init__()
#         """This function implements the LSTM model using Keras and returns the model.

#         Arguments
#         ---------
#         initializer: function
#             The weight initialization function from the torch.nn.init module that is used to initialize
#             the initial weights of the models.
#         vocabulary_size: int
#             The number of words that are to be considered among the words that used most frequently.
#         embedding_size: int
#             The number of dimensions to which the words will be mapped to.
#         hidden_size: int
#             The number of features of the hidden state.
#         dropout: float
#             The dropout rate that will be considered during training.
#         """
#         self.embed = keras.layers.Embedding(input_dim=vocabulary_size,
#                                             output_dim=embedding_size,
#                                             input_length=review_length,
#                                             embeddings_initializer=initializer)
        
#         self.drop = keras.layers.Dropout(dropout)

#         self.lstm = keras.layers.LSTM(hidden_size, kernel_initializer=initializer,
#                                       recurrent_initializer='orthogonal',
#                                       bias_initializer='zeros', use_bias=True)
        
#         self.dense = tf.keras.layers.Dense(units=1, activation='sigmoid')
        

def initialize_keras_lenet5(initializer, dropout):
    
    
    model = keras.models.Sequential([
        keras.layers.Conv2D(6, kernel_size=5, strides=1,  activation='relu', kernel_initializer=initializer, input_shape=(32,32,1)),# dtype=data_type_dict[data_type][framework]), #C1
        keras.layers.MaxPool2D(strides=2), #S2
        keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='relu', kernel_initializer=initializer),# dtype=data_type_dict[data_type][framework]), #C3
        keras.layers.MaxPool2D(strides=2), #S4
        keras.layers.Flatten(), #Flatten
        # keras.layers.Dropout(dropout),
        keras.layers.Dense(120, activation='relu', kernel_initializer=initializer),# dtype=data_type_dict[data_type][framework]), #C5
        keras.layers.Dropout(dropout),
        keras.layers.Dense(84, activation='relu', kernel_initializer=initializer),# dtype=data_type_dict[data_type][framework]), #F6
        keras.layers.Dropout(dropout),
        keras.layers.Dense(10, activation='softmax', kernel_initializer=initializer),# dtype=data_type_dict[data_type][framework]) #Output layer
    ])
    
    
    return model

def keras_training_phase(model, optimizer, loss_fn, X_train_padded, y_train,
                         X_test_padded, y_test, batch_size, n_epochs, device,
                         data_type, experiment):
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print('GPUs already initialized.')
        

    if data_type == 'mixed':
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    if data_type == 'mixed':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

    train_start_timestamp = datetime.datetime.now()
    start = time.time()
    with tf.device(device):
        model.fit(X_train_padded, y_train, batch_size=batch_size,
                             epochs=n_epochs, verbose=1)
    training_time = time.time() - start
    train_end_timestamp = datetime.datetime.now()
    
    start = time.time()
    with tf.device(device):
        accuracy = model.evaluate(X_test_padded, y_test, batch_size=batch_size)[1]
    inference_time = (time.time() - start) / X_test_padded.shape[0]
    
    model.save('./models/lenet5/{}'.format(experiment))
    
    return training_time, inference_time, accuracy * 100.0, train_start_timestamp, train_end_timestamp

def keras_inference_phase(X_test_padded_ext, y_test_ext, batch_size,
                          device, data_type, experiment):

    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        print('GPUs already initialized.')
            
    # if data_type == 'mixed':
    #     optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    if data_type == 'mixed':
        policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(policy)

    model = tf.keras.models.load_model('./models/lenet5/{}'.format(experiment))
    inference_start_timestamp = datetime.datetime.now()
    with tf.device(device):
        accuracy = model.evaluate(X_test_padded_ext, y_test_ext, batch_size=batch_size)[1]
    inference_end_timestamp = datetime.datetime.now()

    print('Accuracy: {}'.format(accuracy))
    return inference_start_timestamp, inference_end_timestamp