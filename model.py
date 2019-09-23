import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


class MLPModel():
    def __init__(self, args):
        self.args = args
        self.feat_dim = args['feat_dim']
        self.batch_size = args['batch_size']
        self.epochs = args['epochs']
        self.lr = args['lr']
        self.build_model()

    def build_model(self):
        inputs = keras.Input(shape=(self.feat_dim,))
        x = keras.layers.Dense(20, activation='relu')(inputs)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(50, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(50, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.Dense(20, activation='relu')(x)
        x = keras.layers.Dropout(0.5)(x)
        outputs = keras.layers.Dense(1)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=tf.train.AdamOptimizer(self.lr),
                           loss='mse',
                           metrics=['mse'])
    
    def train(self, data, label, val_data, val_label):
        data ,label = self.data_scaler(data, label)
        val_data = self.x_scaler.transform(val_data)
        val_label = self.y_scaler.transform(val_label.reshape(-1, 1)).reshape(-1)
        callback = keras.callbacks.EarlyStopping(monitor='val_mse', patience=10, restore_best_weights=True)
        self.model.fit(data, label, batch_size=self.batch_size, epochs=self.epochs, callbacks=[callback], validation_data=(val_data, val_label))

    def data_scaler(self, data, label):
        self.x_scaler = StandardScaler().fit(data)
        scaled_data = self.x_scaler.transform(data)
        self.y_scaler = StandardScaler().fit(label.reshape(-1, 1))
        scaled_label = self.y_scaler.transform(label.reshape(-1, 1)).reshape(-1)
        return scaled_data, scaled_label

    def evaluate(self, test_data, test_label):
        scaled_test_data = self.x_scaler.transform(test_data)
        pred = self.model.predict(scaled_test_data)
        pred_label = pred * self.y_scaler.scale_[0] + self.y_scaler.mean_[0]
        pred_label = pred_label.reshape(-1)
        mae = mean_absolute_error(test_label, pred_label)
        mse = mean_squared_error(test_label, pred_label)
        print('test mae: {}'.format(mae))
        print('test mse: {}'.format(mse))
        return pred_label

# data = np.random.random((100, 1))
# label = data

# val_data = np.random.random((10, 1))
# val_label = val_data

# # x = tf.placeholder(tf.float32, shape=[None, 1], )


# inputs = keras.Input(shape=(1,))
# outputs = keras.layers.Dense(1, 
#                              kernel_initializer=keras.initializers.Constant(1.0),
#                              bias_initializer=keras.initializers.Constant(0.0))(inputs)
# model = keras.Model(inputs=inputs, outputs=outputs)

# model.compile(optimizer=tf.train.AdamOptimizer(0.001),
#               loss='mse',
#               metrics=['mse'])


# model.fit(data, label, epochs=0, batch_size=32,
#           validation_data=(val_data, val_label))

# tf.saved_model.simple_save(keras.backend.get_session(),
#                            "./model",
#                            inputs={'inputs': inputs},
#                            outputs={'outputs': outputs})

# pred_label = model.predict(val_data)
# print('pred label: ', pred_label)
# print('val_label: ', val_label)