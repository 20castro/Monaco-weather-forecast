from formatting.data import Data
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import EmptyDataError
from typing import List
from sklearn.linear_model import LinearRegression
from utils.gaussApprox import approx


class Baseline(Data):

    def __init__(self, obs_path: str, nwp_path: str):
        super().__init__(obs_path, nwp_path)
        self.obs_only = False
        self.labels = None
        self.intercept = None
        self.coef = None

    def base(self, main_variable: str, plot_hist: bool = True, plot_graph: bool = False):

        ''' Compute a simple linear regression on NWP as input and observations as labels '''

        self.n_input = None
        self.n_output = 1
        self.obs_only = False
        self.features = [main_variable]
        self.labels = main_variable

        train_length = 24*153*(3 if self.n_years == 4 else 4)
        test_length = 24*153*(1 if self.n_years == 4 else 2)
        key = 2*self.order[main_variable]

        X_train = self.data[test_length:test_length + train_length, key + 1].reshape(-1, 1)
        X_test = self.data[:test_length, key + 1].reshape(-1, 1)
        y_train = self.data[test_length:test_length + train_length, key].copy()
        y_test = self.data[:test_length, key].copy()

        model = LinearRegression()
        model.fit(X_train, y_train)
        self.intercept = model.intercept_
        self.coef = model.coef_
        y_pred = model.predict(X_test)
        err = y_pred - y_test
        key = 2*self.order[main_variable]
        nwp_err = self.data[:, key + 1] - self.data[:, key]

        self.__plot(y_pred, err, nwp_err, plot_hist, plot_graph)

    def evaluate(
                self,
                main_variable: str,
                other_explanatory_variables: List[str] = [],
                model=LinearRegression(),
                n_input: int = 3,
                normalization: bool = False,
                observation_only: bool = False,
                plot_hist: bool = True,
                plot_graph: bool = False,
                save_baseline: bool = False
            ):

        ''' Provides test using more sophisticated linear regressions than base '''

        date = True if 'date' in other_explanatory_variables else False
        if date: other_explanatory_variables.remove('date')

        if main_variable in other_explanatory_variables:
            other_explanatory_variables.remove(main_variable)
        
        same_labels = self.labels == main_variable
        same_features = self.features == [main_variable] + other_explanatory_variables

        if same_features and same_labels and n_input == self.n_input and observation_only == self.obs_only:
            pass
        else:
            self.__set(main_variable, other_explanatory_variables, n_input, observation_only)

        train, test = self.__get(date, normalization)
        model.fit(train[0], train[1])
        self.intercept = model.intercept_
        self.coef = model.coef_
        y_pred = model.predict(test[0])
        err = y_pred - test[1]
        key = 2*self.order[main_variable]
        nwp_err = self.data[:, key + 1] - self.data[:, key]

        self.__plot(y_pred, err, nwp_err, plot_hist, plot_graph)

        if save_baseline:
            try:
                df = pd.read_csv('baseline.csv', delimiter=';', header='infer', index_col=0)
                mod = pd.read_csv('basemod.csv', delimiter=';', header='infer', index_col=0)
            except (EmptyDataError, FileNotFoundError):
                df = pd.DataFrame()
                mod = pd.DataFrame(index=['model', 'expl. var.', 'input length', 'obs. only'])
            try:
                df[main_variable] = test[1].flatten()
                df[main_variable + '_basepred'] = y_pred.flatten()
            except ValueError:
                print('Wrong length (choose the right dataset or empty baseline.csv) : failure')
            expl_str = main_variable
            for var in other_explanatory_variables: expl_str += ' ' + var
            if date: expl_str += ' date'
            mod[main_variable] = [f'{model}', expl_str, f'{n_input}', f'{observation_only}']
            df.to_csv('baseline.csv', sep=';', header=True, index=True)
            mod.to_csv('basemod.csv', sep=';', header=True, index=True)
            print('Data saved : success')

    def __plot(self, y_pred: np.ndarray, err: np.ndarray, nwp_err: np.ndarray, plot_hist: bool, plot_graph: bool):

        rmse = np.sqrt(np.power(err, 2).mean())
        nwp_rmse = np.sqrt(np.power(nwp_err, 2).mean())
        print(f'Baseline RMSE : {rmse:.2f} | NWP RMSE : {nwp_rmse:.2f}')
        print(f'Baseline MAE  : {np.abs(err).mean():.2f} | NWP MAE  : {np.abs(nwp_err).mean():.2f}')

        n_subplots = int(plot_hist) + int(plot_graph)
        k = 1

        if n_subplots > 0:
            
            fig = plt.figure(figsize=(10, n_subplots*3))

            if plot_hist and self.labels == 'wd':

                plt.subplot(n_subplots, 1, k)

                period_nwp_err = np.concatenate((nwp_err - 360., nwp_err, nwp_err + 360.), axis=None)
                val, key = np.histogram(period_nwp_err, bins=300, density=True)
                val *= 3.
                width = key[1] - key[0]
                plt.bar(key[:-1], val, color='tab:pink', width=width, alpha=.5, label='NWP error distribution')

                period_err = np.concatenate((err - 360., err, err + 360.), axis=None)
                val, key = np.histogram(period_err, bins=300, density=True)
                val *= 3.
                width = key[1] - key[0]
                plt.bar(key[:-1], val, color='tab:cyan', width=width, alpha=.5, label='Baseline error distribution')

                plt.xlim([-365, 365])
                plt.xticks(np.arange(-360, 361, 90))
                plt.legend()
                k += 1

            elif plot_hist:

                ax = plt.subplot(n_subplots, 1, k)
                approx(ax, nwp_err, 'NWP', ('tab:pink', 'tab:purple'), self.labels in ['ws', 'wx', 'wy', 't'])
                approx(ax, err, 'Baseline', ('tab:cyan', 'tab:blue'), self.labels in ['ws', 'wx', 'wy', 't'])
                k += 1

            if plot_graph:

                plt.subplot(n_subplots, 1, k)
                main_variable = self.labels
                super().plot(main_variable, [y_pred], [self.n_input], ['Baseline prediction'])

            fig.tight_layout()
            plt.show()

    def __set(
                self,
                main_variable: str,
                other_explanatory_variables: List[str],
                n_input: int = 3,
                observation_only: bool = False
            ):

        self.n_input = n_input
        self.n_output = 1
        self.obs_only = observation_only

        self.features = [main_variable] + other_explanatory_variables
        self.labels = main_variable

        xslice = np.array([])
        yslice = np.array([])
        for feat_name in self.features:

            if feat_name == main_variable and not observation_only:
                xslice_to_add = np.mod(np.arange(3*n_input), 2*n_input) - n_input
                yslice_to_add = 2*self.order[feat_name] + np.array([0 if k < n_input else 1 for k in range(3*n_input)])
            else:
                xslice_to_add = np.arange(-self.n_input, 0)
                yslice_to_add = 2*self.order[feat_name]*np.ones((self.n_input,), dtype='int')

            xslice = np.concatenate((xslice, xslice_to_add))
            yslice = np.concatenate((yslice, yslice_to_add))

        self.features_slices = xslice.astype('int'), yslice.astype('int')

    def __get(self, date: bool, normalization: bool = True):

        train_length = 24*153*(3 if self.n_years == 4 else 4)
        test_length = 24*153*(1 if self.n_years == 4 else 2)

        train = self.data[test_length:test_length + train_length, :].copy()
        test = self.data[:test_length, :].copy()

        mean = train.mean(axis=0)
        std = train.std(axis=0)

        # Possible values

        start_margin = self.n_input
        end_margin = max(self.n_output, self.n_input) - 1
        
        # Train set

        train_indexes = np.arange(start_margin, train_length - end_margin)
        np.random.shuffle(train_indexes)
        y_train = np.array(
            [train[i:i + self.n_output, 2*self.order[self.labels]] for i in train_indexes]
        )
        if normalization: train = (train - mean)/std
        X_train = np.array(
            [train[i + self.features_slices[0], self.features_slices[1]] for i in train_indexes]
        )

        # Test set

        test_indexes = np.arange(start_margin, test_length - end_margin)
        if self.labels == 'kc':
            test_indexes = test_indexes[self.daylight[start_margin:test_length - end_margin]]
        y_test = np.array(
            [test[i:i + self.n_output, 2*self.order[self.labels]] for i in test_indexes]
        )
        if normalization: test = (test - mean)/std
        X_test = np.array(
            [test[i + self.features_slices[0], self.features_slices[1]] for i in test_indexes]
        )

        if date:

            total_days = len(self.obs['date'][:].data)
            total_hours = len(self.obs['time'][:].data)
            days_train = np.mod(train_indexes, total_days)
            hours_train = np.mod(train_indexes, total_hours)
            days_test = np.mod(test_indexes, total_days)
            hours_test = np.mod(test_indexes, total_hours)
            
            train_add = np.zeros((train_indexes.size, 4))
            train_add[:, 0] = np.cos(2*np.pi*days_train/(365.2425*24))
            train_add[:, 1] = np.sin(2*np.pi*days_train/(365.2425*24))
            train_add[:, 2] = np.cos(2*np.pi*hours_train/24)
            train_add[:, 3] = np.sin(2*np.pi*hours_train/24)
            X_train = np.concatenate((X_train, train_add), axis=1)

            test_add = np.zeros((test_indexes.size, 4))
            test_add[:, 0] = np.cos(2*np.pi*days_test/(365.2425*24))
            test_add[:, 1] = np.sin(2*np.pi*days_test/(365.2425*24))
            test_add[:, 2] = np.cos(2*np.pi*hours_test/24)
            test_add[:, 3] = np.sin(2*np.pi*hours_test/24)
            X_test = np.concatenate((X_test, test_add), axis=1)

        return (X_train, y_train), (X_test, y_test)