from formatting.data import Data
from typing import List
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
import matplotlib.pyplot as plt
from utils.gaussApprox import approx


class MultiTaskShaper(Data):

    def __init__(self, obs_path: str, nwp_path: str):
        super().__init__(obs_path, nwp_path)
        self.cols_desc = {}
        self.labels = []

    def plot(self, feature_name: str, y_pred: np.ndarray, err: np.ndarray):

        rmse = np.sqrt(np.power(err, 2).mean(axis=0))
        mae = np.abs(err).mean(axis=0)
        printed_rmse = 'Model RMSE : '
        printed_mae = 'Model MAE  : '
        for k, (r, m) in enumerate(zip(rmse, mae)):
            if k > 0:
                printed_rmse += ', '
                printed_mae += ', '
            printed_rmse += f'{r:.2f}'
            printed_mae += f'{m:.2f}'

        try:
            basemod = pd.read_csv('models/basemod.csv', header='infer', delimiter=';', index_col=0)
            base = pd.read_csv('models/baseline.csv', header='infer', delimiter=';', index_col=0)
            y_basepred = np.array(base[feature_name + '_basepred']).reshape((-1, 1))
            base_err = y_basepred - np.array(base[feature_name]).reshape((-1, 1))
            base_rmse = np.sqrt(np.power(base_err, 2).mean())
            print(f'Baseline RMSE : {base_rmse:.2f} | ' + printed_rmse)
            print(f'Baseline MAE  : {np.abs(base_err).mean():.2f} | ' + printed_mae)
        except (FileNotFoundError, EmptyDataError, KeyError):
            print(printed_rmse)
            print(printed_mae)
            print('\n----- Baseline not found -----\n')

        fig = plt.figure(figsize=(14, 8))

        # Histogram
        ax = plt.subplot(2, 1, 1)
        try:
            approx(ax, base_err, 'Baseline', ('tab:pink', 'tab:purple'), feature_name in ['ws', 'wx', 'wy', 't'])
        except NameError:
            pass
        for k, e in enumerate(err.T):
            if k%10 == 1: suffix = 'st'
            elif k%10 == 2: suffix = 'nd'
            elif k%10 == 3: suffix = 'rd'
            else: suffix = 'th'
            approx(ax, e, f'{k + 1}' + suffix + ' prediction', ('tab:cyan', 'tab:blue'), feature_name in ['ws', 'wx', 'wy', 't'])
        
        # Graph
        plt.subplot(2, 1, 2)
        try:
            y = [y_basepred, y_pred]
            name_pred = ['Baseline', 'Prediction']
            margin = [int(basemod.loc['input length', feature_name]), self.n_input]
        except (NameError, KeyError):
            name_pred = ['Prediction']
            y = [y_pred]
            margin = [self.n_input]
        super().plot(feature_name, y, margin, name_pred)

        fig.tight_layout()

    def error_comparison(self, feature_name: str, y_pred: np.ndarray):
        y = y_pred[:, 0]
        nwp = self.data[self.n_input:self.n_input + y.size, 2*self.order[feature_name] + 1]
        obs = self.data[self.n_input:self.n_input + y.size, 2*self.order[feature_name]]
        plt.figure()
        plt.grid()
        plt.scatter((y - obs).flatten(), (nwp - obs).flatten(), alpha=.4, marker='+', s=2)
        plt.xlabel('Model error')
        plt.ylabel('NWP error')
        plt.show()
        
    def __set(self, features: List[str], labels: List[str], n_input: int = 3, n_output: int = 1):

        self.features = features
        self.labels = labels
        self.n_input = n_input
        self.n_output = n_output
        self.cols_desc = {}
        key = 0

        xslice = np.array([])
        yslice = np.array([])
        for feat_name in features:

            self.cols_desc[feat_name + '_past_obs'] = key
            key += 1

            if feat_name in labels:
                xslice_to_add = np.mod(np.arange(3*n_input), 2*n_input) - n_input
                yslice_to_add = 2*self.order[feat_name] + np.array([0 if k < n_input else 1 for k in range(3*n_input)])
                self.cols_desc[feat_name + '_next_nwp'] = key
                key += 1
                self.cols_desc[feat_name + '_past_nwp'] = key
                key += 1
            else:
                xslice_to_add = np.arange(-self.n_input, 0)
                yslice_to_add = 2*self.order[feat_name]*np.ones((self.n_input,), dtype='int')

            xslice = np.concatenate((xslice, xslice_to_add))
            yslice = np.concatenate((yslice, yslice_to_add))

        self.features_slices = xslice.astype('int'), yslice.astype('int')

    def __shapeX(self, X):
        ''' Shape X_train set into (batch size, number of features types, number of timesteps per feature) '''
        return X.reshape((X.size // self.n_input, self.n_input)).T

    def get(self, features: List[str], labels: List[str], n_input: int = 3, n_output: int = 1):

        '''

        Returns asked data

        Args:
        - features : List[str] containing wanted explanatory features' names
        - labels : List[str] containing the names of features targeted by the prediction
        - n_input : int corresponding to the number of timesteps used to feed to model per feature
        - n_output : int corresponding to the number of timesteps we want to predict

        Returns :
        * 3 tuples (X, y) corresponding resp. to the train, test and validation (features, labels) sets
        * Caution : the y are lists containing as many elements as labels (necessary to multi-task training)
        * Each element of y is a np.ndarray of shape (n_samples, n_output) corressponding to one label

        '''
        
        same_features = features == self.features
        same_labels = labels == self.labels
        same_input = n_input == self.n_input
        same_output = n_output == self.n_output

        if same_features and same_labels and same_input and same_output:
            pass
        else:
            self.__set(features, labels, n_input, n_output)

        train_length = 24*153*(2 if self.n_years == 4 else 3)
        validation_length = 24*153
        test_length = 24*153*(1 if self.n_years == 4 else 2)

        train = self.data[test_length:test_length + train_length, :].copy()
        validation = self.data[test_length + train_length:, :].copy()
        test = self.data[:test_length, :].copy()

        mean = train.mean(axis=0)
        std = train.std(axis=0)

        # Possible values

        start_margin = self.n_input
        end_margin = max(self.n_output, self.n_input) - 1
        
        # Train set

        train_indexes = np.arange(start_margin, train_length - end_margin)
        np.random.shuffle(train_indexes)
        y_train = [
            np.array([train[i:i + self.n_output, 2*self.order[lab]] for i in train_indexes]) for lab in self.labels
        ]
        train = (train - mean)/std
        X_train = np.array(
            [
                self.__shapeX(
                    train[i + self.features_slices[0], self.features_slices[1]]
                ) for i in train_indexes
            ]
        )

        # Validation set

        validation_indexes = np.arange(start_margin, validation_length - end_margin)
        np.random.shuffle(validation_indexes)
        y_validation = [
            np.array([validation[i:i + self.n_output, 2*self.order[lab]] for i in validation_indexes]) for lab in self.labels
        ]
        validation = (validation - mean)/std
        X_validation = np.array(
            [
                self.__shapeX(
                    validation[i + self.features_slices[0], self.features_slices[1]]
                ) for i in validation_indexes
            ]
        )

        # Test set

        test_indexes = np.arange(start_margin, test_length - end_margin)
        if 'kc' in self.labels:
            test_indexes = test_indexes[self.daylight[start_margin:test_length - end_margin]]
        y_test = [
            np.array([test[i:i + self.n_output, 2*self.order[lab]] for i in test_indexes]) for lab in self.labels
        ]
        test = (test - mean)/std
        X_test = np.array(
            [
                self.__shapeX(
                    test[i + self.features_slices[0], self.features_slices[1]]
                ) for i in test_indexes
            ]
        )

        return (X_train, y_train), (X_test, y_test), (X_validation, y_validation)

    def get_residual_indexes(self):

        indexes = []

        if self.n_input < self.n_output:
            raise ValueError("The number of output timesteps exceeds the input's")

        for lab in self.labels:
            try:
                indexes.append(self.cols_desc[lab + '_next_nwp'])
            except KeyError:
                # Not available labels (ie : those not chosen among features which are
                # thus not eligible to residuals) will have the model compute them standardly
                indexes.append(None)

        return indexes
