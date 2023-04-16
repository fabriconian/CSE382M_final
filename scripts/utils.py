import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import os
import sklearn as sk
from keras import metrics
from keras import callbacks
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model
import time
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.random_projection import GaussianRandomProjection, \
    SparseRandomProjection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from tqdm import tqdm

class DimReductionEvaluation:
    def __init__(self, data=None):
        # data in the form of a pandas dataframe with the following columns:
        #     - image: the image file
        #     - label: the label of the image
        self.data = data
        
    def load_data(self, folder_path, image_size):
        # specify the shape of the resized image
        shape = (image_size, image_size)  
        self.data_folder=folder_path
        self.image_size = image_size
        self.image_shape = shape
        # Create an empty dictionary to store the data
        data = {'image': [], 'label': [], 'numpy': [], 
                'numpy_grayscale': [], 'numpy_flaten': []}

        # loop through each category folder
        for category in os.listdir(folder_path):
            category_path = os.path.join(folder_path, category)
            if os.path.isdir(category_path):
                # loop through each image file in the category folder
                for file in os.listdir(category_path):
                    file_path = os.path.join(category_path, file)
                    if os.path.isfile(file_path):
                        # open the image file and resize it
                        with Image.open(file_path) as image:
                            label = category
                            # Add the image and label to the dictionary
                            data['image'].append(image)
                            data['label'].append(label)
                            data['numpy_flaten'].append(np.array(image.resize(shape)).reshape(-1)/127.5-1)
                            data['numpy'].append(np.array(image.resize(shape))/127.5-1.0)
                            data['numpy_grayscale'].append(np.array(image.resize(shape)).mean(axis=-1).reshape(-1)/255)
        # Convert the dictionary to a pandas dataframe
        # data['numpy_normalized']=list(sk.preprocessing.normalize(np.array(data['numpy'])))
        data['numpy_normalized']=\
            list(sk.preprocessing.normalize(np.array(data['numpy_flaten'])))
        data['numpy_grayscale_normalized']=\
            list(sk.preprocessing.normalize(np.array(data['numpy_grayscale'])))
        self.dim=data['numpy'][0].reshape(-1).shape[0]
        self.data = pd.DataFrame(data)
        self.X=np.array([x for x in self.data['numpy'].to_numpy()])
        self.X_flaten=np.array([x for x 
                    in self.data['numpy_flaten'].to_numpy()])
        
        self.X_flaten_grayscale=np.array([x for x 
                    in self.data['numpy_grayscale_normalized'].to_numpy()])
        
        self.Y=self.data['label'].to_numpy()
    
    def split_data(self, x_data, y_data, test_size=0.1):
        x_train, x_test, y_train, y_test=train_test_split(x_data, y_data,
                test_size=test_size)
        return x_train, x_test, y_train, y_test
        
    def make_datagen(self, **kwargs):
        if not kwargs is None:
            self.datagen=tf.keras.preprocessing.image.ImageDataGenerator(**kwargs)
        else:
            self.datagen=tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.2, # Randomly zoom image 
                shear_range=0.2,
                preprocessing_function=lambda x: x/127.5-1.0,
                # rescale=1./255,         # rescale it (I guess for the better numbers for the gradients (no blow ups))
                validation_split=0.2,  # split the dataset into train and validation parts
                width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)
            
    def get_train_data(self, batch_size=32):
        return self.datagen.flow_from_directory(self.data_folder,
                                                 batch_size=batch_size,
                                                 target_size=self.image_shape,
                                                 class_mode='categorical',
                                                 subset='training')
    
    def get_validation_data(self, batch_size=32):
        return self.datagen.flow_from_directory(self.data_folder,
                                                 batch_size=batch_size,
                                                 target_size=self.image_shape,
                                                 class_mode='categorical',
                                                 subset='validation')

class DAE:
    def __init__(self, input_dim, batch_size, latent_dim):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        input_img = Input(shape=(self.input_dim, ))

        # 'encoded' is the encoded representation of the input
        encoded = Dense(
            int(self.input_dim / 2),
            kernel_initializer='glorot_uniform')(input_img)

        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(
            int(self.input_dim / 4),
            kernel_initializer='glorot_uniform')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(
            int(self.input_dim / 8),
            kernel_initializer='glorot_uniform')(encoded)
        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(self.latent_dim, activation='linear')(encoded)
        # 'decoded' is the lossy reconstruction of the input
        decoded = Dense(
            int(self.input_dim / 8),
            kernel_initializer='glorot_uniform')(encoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            int(self.input_dim / 4),
            kernel_initializer='glorot_uniform')(decoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            int(self.input_dim / 2),
            kernel_initializer='glorot_uniform')(decoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            self.input_dim,
            activation='sigmoid',
            kernel_initializer='glorot_uniform')(decoded)

        self.autoencoder = Model(inputs=input_img, outputs=decoded)
        self.autoencoder.compile(optimizer='Adam', loss='mse')
        self.encoder = Model(inputs=input_img, outputs=encoded)

    # return a fit deep encoder
    def fit(self, x_train, y_train, epochs=9999, verbose=0):
        self.x_train, self.x_valid = model_selection.train_test_split(
            x_train,
            test_size=int(
                0.1 * x_train.shape[0] // self.batch_size * self.batch_size),
            train_size=int(
                0.9 * x_train.shape[0] // self.batch_size * self.batch_size),
            stratify=y_train)

        history=self.autoencoder.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(self.x_valid, self.x_valid),
            verbose=verbose,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.01,
                    patience=5,
                    restore_best_weights=True)
            ])
        return history

    # return prediction for x
    def transform(self, x):
        prediction = self.encoder.predict(x)
        return prediction.reshape((len(prediction),
                                   np.prod(prediction.shape[1:])))

class DAE_CNN:
    def __init__(self, batch_size, input_shape, latent_dim):
        input_img = Input(shape=input_shape)
        self.latent_dim = latent_dim
        if input_shape == (64, 64, 3):
            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same', strides=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim, activation='sigmoid'),
                tf.keras.layers.BatchNormalization()
            ])
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.latent_dim,)),
                tf.keras.layers.Dense(8*8*256, activation='relu'),
                tf.keras.layers.Reshape((8, 8, 256)),
                tf.keras.layers.Conv2DTranspose(128, (3,3), activation='relu', padding='same', strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(64, (3,3), activation='relu', padding='same', strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(32, (3,3), activation='relu', padding='same', strides=2),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2DTranspose(3, (3,3), activation='sigmoid', padding='same')
            ])
            
        elif input_shape == (32, 32, 3):

            # self.encoder = tf.keras.Sequential([
            #     tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
            #         padding='same', input_shape=input_shape),
            #     tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Flatten(),
            #     tf.keras.layers.Dense(self.latent_dim, activation='linear'),
            #     # tf.keras.layers.BatchNormalization(),
            # ])

            # # Define the decoder model
            # self.decoder = tf.keras.Sequential([
            #     tf.keras.layers.Dense(256, activation='relu', input_shape=(self.latent_dim,)),
            #     tf.keras.layers.Reshape((1, 1, 256)),
            #     tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.UpSampling2D((2, 2)),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.UpSampling2D((2, 2)),
            #     tf.keras.layers.BatchNormalization(),  
            #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.UpSampling2D((2, 2)),
            #     tf.keras.layers.BatchNormalization(),              
            #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.UpSampling2D((2, 2)),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            #     tf.keras.layers.UpSampling2D((2, 2)),
            #     tf.keras.layers.BatchNormalization(),
            #     tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
            # ])

            self.encoder = tf.keras.Sequential([
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                    padding='same', input_shape=input_shape),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.MaxPooling2D((2, 2), padding='same'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(self.latent_dim, activation='linear'),
               # tf.keras.layers.BatchNormalization()
            ])

            # Define the decoder model
            self.decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(8*8*32, activation='relu', input_shape=(self.latent_dim,)),
                tf.keras.layers.Reshape((8, 8, 32)),
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
                tf.keras.layers.UpSampling2D((2, 2)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
            ])
        else:
            raise ValueError("Invalid input shape. Must be either (64, 64, 3) or (32, 32, 3).")
        
        self.autoencoder = tf.keras.Model(input_img, self.decoder(self.encoder(input_img)))
        self.autoencoder.compile(optimizer='Adam', loss='mse')
        self.batch_size = batch_size

    def fit(self, x_train, y_train, epochs=9999, verbose=0):
        self.x_train, self.x_valid = model_selection.train_test_split(
            x_train,
            test_size=int(
                0.1 * x_train.shape[0] // self.batch_size * self.batch_size),
            train_size=int(
                0.9 * x_train.shape[0] // self.batch_size * self.batch_size),
            stratify=y_train)

        History=self.autoencoder.fit(
            x_train,
            x_train,
            epochs=epochs,
            batch_size=self.batch_size,
            validation_data=(self.x_valid, self.x_valid),
            verbose=verbose,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.01,
                    patience=5,
                    restore_best_weights=True)
            ])
        return History

    # return prediction for x
    def transform(self, x):
        prediction = self.encoder.predict(x)
        return prediction
    
    # def fit(self, data, epochs=10, batch_size=32, learning_rate=0.001):
    #     self.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    #     self.fit(data, data, epochs=epochs, batch_size=batch_size)

class PCA:
    def __init__(self, x_train, n_components_max, fit=True):
        self.pca = sk.decomposition.PCA(n_components=n_components_max)

        self.pca.fit(x_train)
    
    def transform(self, x, k):
        return self.pca.transform(x)[:,:k]
    
class ReductionAnalysis:
    def __init__(self, X, Y, k_range, batch_size=32):
        self.X = X
        self.Y = Y
        self.flat_dim=self.X[0].reshape(-1).shape[0]
        self.k_range = k_range
        self.n_components_max = k_range[-1]
        self.batch_size = batch_size

        self.params = {
            "n_neighbors":
            [i for i in range(1, int(np.sqrt(self.flat_dim)))]
        }
        self.random_search = RandomizedSearchCV(
            KNeighborsClassifier(),
            param_distributions=self.params,
            n_iter=60,
            cv=5,
            n_jobs=-1)
        self.split_data()
        self.zero_results()

    def set_k_range(self, k_range):
        self.k_range = k_range
        self.calculate_linear_projections()

    def split_data(self):    
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.Y, test_size=0.1)
        self.x_train_flat=self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_test_flat=self.x_test.reshape(self.x_test.shape[0], -1) 
        self.calculate_linear_projections()
        
    def calculate_linear_projections(self):
        # calculate pca and time it
        start = time.time()
        self.pca = sk.decomposition.PCA(n_components=self.n_components_max)
        self.pca.fit(self.x_train_flat)
        self.pca_train_projections = self.pca.transform(self.x_train_flat)
        self.pca_test_projections = self.pca.transform(self.x_test_flat)
        self.pca_time = time.time() - start

        # calculate srp and time it
        start = time.time()
        self.srp=SparseRandomProjection(n_components=self.n_components_max)
        self.srp_train_projection=self.srp.fit_transform(self.x_train_flat)
        self.srp_test_projection=self.srp.transform(self.x_test_flat)
        self.srp_time=time.time()-start

        # calculate rp and time it
        start = time.time()
        self.rp=GaussianRandomProjection(n_components=self.n_components_max)
        self.rp_train_projection=self.rp.fit_transform(self.x_train_flat)
        self.rp_test_projection=self.rp.transform(self.x_test_flat)
        self.rp_time=time.time()-start

    def evaluate(self, methods, verbose=0):
        for k in tqdm(self.k_range):
            for method in methods:
                new_df = pd.DataFrame(self.evaluate_k(method, k, verbose=verbose),
                         index=[0])
                self.results = pd.concat([self.results, new_df], ignore_index=True)
                # self.results.append(self.evaluate_k(method, k, verbose=verbose))

    def zero_results(self):
        self.results = pd.DataFrame({'method': [], 'score': [], 'time': [], 'k': []})

    def evaluate_k(self, method, k, verbose=0):
        if method == 'PCA':
            self.random_search.fit(self.pca_train_projections[:,:k], 
                self.y_train)
            score=self.random_search.score(self.pca_test_projections[:,:k], 
                self.y_test)
            t_time=self.pca_time
        elif method == 'SRP':   
            self.random_search.fit(self.srp_train_projection[:,:k], 
                self.y_train)
            score=self.random_search.score(self.srp_test_projection[:,:k], 
                self.y_test)
            t_time=self.srp_time
        elif method == 'RP':
            self.random_search.fit(self.rp_train_projection[:,:k], 
                self.y_train)
            score=self.random_search.score(self.rp_test_projection[:,:k], 
                self.y_test)
            t_time=self.rp_time
        elif method == 'DAE':
            start=time.time()
            self.dae=DAE(latent_dim=k, batch_size=self.batch_size,
                input_dim=self.x_train_flat.shape[1])
            self.dae.fit(self.x_train_flat, self.y_train,
                epochs=100, verbose=verbose)
            train_projections=self.dae.transform(self.x_train_flat)
            test_projections=self.dae.transform(self.x_test_flat)
            t_time=time.time()-start

            self.random_search.fit(train_projections, self.y_train)
            score=self.random_search.score(test_projections, self.y_test)
        elif method == 'DAE_CNN':
            start=time.time()
            self.dae=DAE_CNN(latent_dim=k, batch_size=self.batch_size,
                input_shape=self.x_train[0].shape)
            self.dae.fit(self.x_train, self.y_train,
                epochs=100, verbose=verbose)
            train_projections=self.dae.transform(self.x_train)
            test_projections=self.dae.transform(self.x_test)
            t_time=time.time()-start

            self.random_search.fit(train_projections, self.y_train)
            score=self.random_search.score(test_projections, self.y_test)
        
        return {'method': method, 'score': score, 'time': t_time, 'k': k}


