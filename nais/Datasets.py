import pandas as pd

import tarfile
import numpy as np
import h5py
import os
from tqdm import tqdm

from random import choices

from sklearn.model_selection import train_test_split

import getpass
user = getpass.getuser()


class Dataset:
    def __init__(self, location=None, target_location=None):
        #load data into memory or create generator if large.
        self.location = location
        self.filename = location.split('/')[-1]
        self.target_location = target_location

        if not os.path.isdir(target_location):
            os.mkdir(target_location)
        if not self.processed:
            self._download_and_uncompress()

    def _download_and_uncompress(self):
        print('Downloading data.')
        f = tarfile.open(self.location)
        f.extractall(self.target_location)
        f.close()

    def get_X_y(self):
        raise NotImplementedError

    def stats(self):
        raise NotImplementedError

class Arces(Dataset):
    def __init__(self, processed = True):
        """
        Loads ARCES dataset.

        processed : boolean
            Set to False to force reload of data.
        """
        if processed == True:
            self.processed = os.path.isdir(f'/nobackup/{user}/tmpdata/arces_data/')
        else:
            self.processed = processed
        super(Arces, self).__init__(location='/projects/processing/ML/ARCES.tar.gz', target_location=f'/nobackup/{user}/tmpdata/')

    def get_X_y(self, include_info=False, subsample=1.0):
        """
        Get X and y.
            include_info : boolean
                include trace information.

        return
            np.array, list
            or
            np.array, list, list if include_info=True
        """
        data = pd.read_csv(self.target_location+'csv_folder/pure_events.csv', header=None, names=['filename', 'class'])
        data['filename'] = data['filename'].apply(lambda f: os.path.join(*f.split('\\')))
        y = data['class'].values

        if subsample < 1:
            _, filenames = train_test_split(data['filename'].values, test_size=subsample, stratify=data['class'].values)
        else:
            filenames = data['filename'].values

        d = [self._load_single(self.target_location + f) for f in tqdm(filenames, desc='Loading traces.')]
        x, info = zip(*d)
        x = np.swapaxes(np.asarray(x), 1, 2)

        if include_info:
            return x, y, info
        else:
            return x, y

    def _load_single(self, filename):
        with h5py.File(filename,'r') as dp:
            trace_array = np.array(dp.get('traces'))
            info = np.array(dp.get('event_info'))
        return trace_array, info
