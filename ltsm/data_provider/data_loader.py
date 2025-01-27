import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

from torch.utils.data.dataset import ConcatDataset, Dataset

from ltsm.utils.timefeatures import time_features
from ltsm.utils.tools import convert_tsf_to_dataframe

warnings.filterwarnings('ignore')

class HF_Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __read_data__(self):
        return self.dataset.__read_data__()

    def __len__(self):
        return self.dataset.__len__()

    def inverse_transform(self, data):
        return self.dataset.inverse_transform(data)

    def add_data(self, df):
        return self.dataset.add_data(df)

    def __getitem__(self, index): 
        outputs = self.dataset.__getitem__(index)
        seq_x = outputs[0]
        seq_y = outputs[1]

        return {
            "input_data": seq_x,
            "labels": seq_y
        }
    
class HF_Timestamp_Dataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __read_data__(self):
        return self.dataset.__read_data__()

    def __len__(self):
        return self.dataset.__len__()

    def inverse_transform(self, data):
        return self.dataset.inverse_transform(data)

    def add_data(self, df):
        return self.dataset.add_data(df)

    def __getitem__(self, index): 
        seq_x, seq_y, seq_x_mark, seq_y_mark = self.dataset.__getitem__(index)

        return {
            "input_data": seq_x,
            "labels": seq_y,
            "timestamp_input": seq_x_mark,
            "timestamp_labels": seq_y_mark
        }

class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        data_path,
        split='train',
        size=None,
        features='S',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h', 
        percent=100,
        max_len=-1,
        train_all=False,
    ):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.pred_len = size
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.percent = percent
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        print("self.enc_in = {}".format(self.enc_in))
        print("self.data_x = {}".format(self.data_x.shape))
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.enc_in > 1:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        else:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        data_path,
        split='train',
        size=None,
        features='S',
        target='OT',
        scale=True,
        timeenc=0,
        freq='t', 
        percent=100,
        max_len=-1,
        train_all=False
    ):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.pred_len = size
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.data_path = data_path
        self.__read_data__()

        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        if self.enc_in > 1:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        else:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
            
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(
        self,
        data_path,
        split='train',
        size=None,
        features='S',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        percent=10,
        max_len=-1,
        train_all=False
    ):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.pred_len = size
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent

        self.data_path = data_path
        self.__read_data__()
        
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.set_type == 0:
            border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        if self.enc_in > 1:
            seq_x = self.data_x[s_begin:s_end]
            seq_y = self.data_y[r_begin:r_end]
        else:
            seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
            seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Pred(Dataset):
    def __init__(
        self,
        data_path,
        split='pred',
        size=None,
        features='S',
        target='OT',
        scale=True,
        inverse=False,
        timeenc=0,
        freq='15min',
        cols=None,
        percent=None,
        train_all=False,
    ):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.pred_len = size
        # init
        assert split in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin]
        else:
            seq_y = self.data_y[r_begin:r_begin]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_TSF(Dataset):
    def __init__(self,
        data_path,
        split='train',
        size=None,
        features='S',
        target='OT',
        scale=True,
        timeenc=0,
        freq='Daily',
        percent=10,
        max_len=-1,
        train_all=False,
    ):
        
        self.train_all = train_all
        
        self.seq_len = size[0]
        self.pred_len = size[2]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]
        
        self.percent = percent
        self.max_len = max_len
        if self.max_len == -1:
            self.max_len = 1e8

        self.data_path = data_path

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.__read_data__()
        
        self.tot_len = self.len_index[-1]


    def __read_data__(self):
        self.scaler = StandardScaler()
        df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(self.data_path)
        self.freq = frequency
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        
        self.data_all = []
        self.len_index = [0]
        self.tot_len = 0
        for i in range(len(timeseries)):
            df_raw = timeseries[i].reshape(-1, 1)

            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        
            if self.set_type == 0:
                border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

            if self.scale:
                train_data = df_raw[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(df_raw)
            else:
                data = df_raw

            self.data_all.append(data[border1:border2])
            self.len_index.append(self.len_index[-1] + border2 - border1 - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        i = 0
        for i in range(len(self.len_index)):
            if index < self.len_index[i]:
                i -= 1
                break
        s_begin = index - self.len_index[i]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_all[i][s_begin:s_end]
        seq_y = self.data_all[i][r_begin:r_end]

        return seq_x, seq_y, np.empty(shape=(self.seq_len, 0)), np.empty(shape=(self.pred_len, 0))

    def __len__(self):
        return self.tot_len

class Dataset_Custom_List(Dataset):
    def __init__(
        self,
        data_path=[],
        split='train',
        size=None,
        features='M',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        percent=10,
        max_len=-1,
        train_all=False
    ):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.pred_len = size
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.data_path = data_path
        self.__read_data__()
        
        self.tot_len = self.len_index[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data_all = []
        self.len_index = [0]
        self.tot_len = 0
        for path in self.data_path:
            if path.endswith('.csv'):
                df_raw = pd.read_csv(path)
            elif path.endswith('.feather'):
                df_raw = pd.read_feather(path)
            df_raw = df_raw.dropna()
            df_raw = df_raw.values
            if self.scale:
                df_raw = self.scaler.fit_transform(df_raw)
            self.data_all.append(df_raw)
            self.len_index.append(self.len_index[-1] + len(df_raw) - self.seq_len - self.pred_len + 1)

    def __getitem__(self, index):
        i = 0
        for i in range(len(self.len_index)):
            if index < self.len_index[i]:
                i -= 1
                break
        s_begin = index - self.len_index[i]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_all[i][s_begin:s_end]
        seq_y = self.data_all[i][r_begin:r_end]

        return seq_x, seq_y, np.empty(shape=(self.seq_len, 0)), np.empty(shape=(self.pred_len, 0))

    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Custom_List_TS(Dataset):
    def __init__(
        self,
        data_path=[],
        split='train',
        size=None,
        features='M',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        percent=10,
        max_len=-1,
        train_all=False
    ):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, self.pred_len = size
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.data_path = data_path
        self.__read_data__()
        
        self.tot_len = self.len_index[-1]

    def __read_data__(self):
        self.scaler = StandardScaler()
        self.data_all = []
        self.len_index = [0]
        self.tot_len = 0

        for path in self.data_path:
            if path.endswith('.csv'):
                df_raw = pd.read_csv(path)
            elif path.endswith('.feather'):
                df_raw = pd.read_feather(path)
            # df_raw = df_raw.dropna()
            # df_raw = df_raw.values

            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        
            if self.set_type == 0:
                border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

            if self.scale:
                train_data = df_raw[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_raw.values)
            else:
                data = df_raw.values

            self.data_all.append(data[border1:border2])
            self.len_index.append(self.len_index[-1] + border2 - border1 - self.seq_len - self.pred_len + 1)

    def add_data(self, df):
        assert len(df) >= self.seq_len + self.pred_len
        self.data_all.append(df)
        self.len_index.append(self.len_index[-1] + len(df) - self.seq_len - self.pred_len + 1)
        self.tot_len = self.len_index[-1]

    def __getitem__(self, index):
        i = 0
        for i in range(len(self.len_index)):
            if index < self.len_index[i]:
                i -= 1
                break
        s_begin = index - self.len_index[i]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_all[i][s_begin:s_end]
        seq_y = self.data_all[i][r_begin:r_end]

        return seq_x, seq_y, np.empty(shape=(self.seq_len, 0)), np.empty(shape=(self.pred_len, 0))

    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom_List_TS_TSF(Dataset):
    def __init__(
        self,
        data_path=[],
        split='train',
        size=None,
        features='M',
        target='OT',
        scale=True,
        timeenc=0,
        freq='h',
        percent=10,
        max_len=-1,
        train_all=False
    ):
        # size [seq_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len, _, self.pred_len = size
        # init
        assert split in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[split]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.percent = percent
        self.data_path = data_path
        self.__read_data__()
        
        self.tot_len = self.len_index[-1]
        
    def __read_data__(self):
        self.scaler = StandardScaler()
        def dropna(x):
            return x[~np.isnan(x)]
        self.data_all = []
        self.len_index = [0]
        self.tot_len = 0

        for path in self.data_path:
            df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(path)
            self.freq = frequency
            timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
            
            for timeserie in timeseries:
                df_raw = timeserie.reshape(-1, 1)

                num_train = int(len(df_raw) * 0.7)
                num_test = int(len(df_raw) * 0.2)
                num_vali = len(df_raw) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]
            
                if self.set_type == 0:
                    border2 = (border2 - self.seq_len) * self.percent // 100 + self.seq_len

                if self.scale:
                    train_data = df_raw[border1s[0]:border2s[0]]
                    self.scaler.fit(train_data)
                    data = self.scaler.transform(df_raw)
                else:
                    data = df_raw

                self.data_all.append(data[border1:border2])
                self.len_index.append(self.len_index[-1] + border2 - border1 - self.seq_len - self.pred_len + 1)

    def add_data(self, df):
        assert len(df) >= self.seq_len + self.pred_len
        self.data_all.append(df)
        self.len_index.append(self.len_index[-1] + len(df) - self.seq_len - self.pred_len + 1)
        self.tot_len = self.len_index[-1]

    def __getitem__(self, index):
        i = 0
        for i in range(len(self.len_index)):
            if index < self.len_index[i]:
                i -= 1
                break
        s_begin = index - self.len_index[i]
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = self.data_all[i][s_begin:s_end]
        seq_y = self.data_all[i][r_begin:r_end]

        return seq_x, seq_y, np.empty(shape=(self.seq_len, 0)), np.empty(shape=(self.pred_len, 0))

    def __len__(self):
        return self.tot_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)