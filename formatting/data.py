import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from typing import List, Optional

class Data:

    def __findIndex(self, lat, lon):
        i = np.argmin(np.abs(self.nwp['lat'][:].data - lat))
        j = np.argmin(np.abs(self.nwp['lon'][:].data - lon))
        return i, j

    def __init__(self, obs_path: str, nwp_path: str):
        self.obs = nc.Dataset(obs_path)
        self.nwp = nc.Dataset(nwp_path)
        self.length = len(self.obs['date'][:].data)*len(self.obs['time'][:].data)
        self.n_years = self.length//(153*24)
        self.i, self.j = self.__findIndex(43.38, 7.83)
        self.order = {'ghi': 0, 't': 1, 'ws': 2, 'wd': 3, 'kc': 4, 'wx': 5, 'wy': 6}
        self.features = []
        self.data = self.__collect()
        self.n_input = None
        self.n_output = None

    def __collect(self):

        ''' Parses the netCDF files targeted into a numpy array '''

        features = ['ghi', 't', 'ws', 'wd']
        data = np.zeros((self.length, 14), dtype='float')

        for k, feat_name in enumerate(features):
            data[:, 2*k] = self.__gapfill(
                self.obs[feat_name][:].reshape((self.length,)),
                feat_name
            )
            data[:, 2*k + 1] = self.__gapfill(
                self.nwp[feat_name][:, 1:, self.i, self.j].reshape((self.length,)),
                feat_name
            )
        
        cs = self.obs['cs'][:].reshape((self.length,))
        self.daylight = cs > 4
        data[self.daylight, 8] = data[self.daylight, 2*self.order['ghi']]/cs[self.daylight]
        data[self.daylight, 9] = data[self.daylight, 2*self.order['ghi'] + 1]/cs[self.daylight]

        data[:, 10] = data[:, 4]*np.cos(np.pi*data[:, 6]/180)
        data[:, 11] = data[:, 5]*np.cos(np.pi*data[:, 7]/180)
        data[:, 12] = data[:, 4]*np.sin(np.pi*data[:, 6]/180)
        data[:, 13] = data[:, 5]*np.sin(np.pi*data[:, 7]/180)

        return data

    def __gapfill(self, column: np.ndarray, name: str):

        ''' Fill missing values '''

        nanloc = np.isnan(column)
        if nanloc.any():
            rean = self.obs[name + '_rean'][:].reshape((self.length,))
            for k in np.arange(self.length)[nanloc]:
                if k == 0 or k == self.length - 1 or nanloc[k - 1] or nanloc[k + 1]:
                    column[k] = rean[k]
                else:
                    column[k] = .5*(column[k - 1] + column[k + 1])
        return column

    def frame(self, features: Optional[List[str]] = None):

        ''' Returns the data as a pd.DataFrame'''

        col_id = []
        cols = []
        feat = self.order.keys() if features is None else features
        for f in feat:
            try:
                id = 2*self.order[f]
            except KeyError:
                print(f'KeyError : feature {f} not valid')
            col_id.append(id)
            col_id.append(id + 1)
            cols.append(f)
            cols.append(f + '_nwp')
        df = pd.DataFrame(data=self.data[:, col_id], columns=cols)
        return df

    def plot(self, feature_name: str, y_pred: List[np.ndarray], shift: Optional[List[int]] = None, name_pred: Optional[List[str]] = None):

        ''' Plot predictions against NWP and measured values '''

        if len(y_pred) > 5:
            raise IndexError("You're trying to plot more than 5 series, that doesn't sound like a good idea !")

        if name_pred is None:
            name_pred = [f'Prediction {k + 1}' for k in range(len(y_pred))]
        if shift is None:
            shift = [0 for _ in y_pred]

        color = ['tab:blue', 'tab:purple', 'tab:gray', 'tab:red', 'tab:black']
        length = 96
        time = np.arange(length)
        beg = max(shift) # assumes test_indexes are at the beginning of the dataset (which is the case)
        ind = 2*self.order[feature_name]

        plt.plot(
            time,
            self.data[beg:beg + length, ind],
            label='Observations',
            color='tab:cyan'
        )
        plt.plot(
            time,
            self.data[beg:beg + length, ind + 1],
            label='NWP',
            color='tab:pink'
        )
        
        k = 0
        for i, y in enumerate(y_pred):
            start_i = max(shift) - shift[i]
            for j, prev in enumerate(y.T):
                plt.scatter(
                    time + j,
                    prev[start_i:start_i + length],
                    label=name_pred[i] + f' with shift {j + 1}',
                    marker='+',
                    color=color[k]
                )
                k += 1
        
        plt.legend()
        plt.xlabel('Time (h)')
