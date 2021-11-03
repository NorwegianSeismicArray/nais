import pandas as pd

import tarfile
import numpy as np
import h5py
import os
from tqdm import tqdm

from random import choices

from sklearn.model_selection import train_test_split
from shutil import copyfile
from PIL import Image

import getpass

user = getpass.getuser()


class Dataset:
    def __init__(self, location=None, target_location=None):
        # load data into memory or create generator if large.
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
    def __init__(self, processed=True):
        """
        Loads ARCES dataset.

        processed : boolean
            Set to False to force reload of data.
        """
        if processed == True:
            self.processed = os.path.isdir(f'/nobackup/{user}/tmpdata/arces_data/')
        else:
            self.processed = processed
        super(Arces, self).__init__(location='/projects/processing/ML/ARCES.tar.gz',
                                    target_location=f'/nobackup/{user}/tmpdata/arces/')

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
        data = pd.read_csv(self.target_location + 'csv_folder/pure_events.csv', header=None,
                           names=['filename', 'class'])
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
        with h5py.File(filename, 'r') as dp:
            trace_array = np.array(dp.get('traces'))
            info = np.array(dp.get('event_info'))
        return trace_array, info


class CoSSen(Dataset):
    def __init__(self):
        """
        Loads CoSSen dataset.
        """
        self.processed = True
        super(CoSSen, self).__init__(location='/projects/processing/ML/cossen/',
                                     target_location=f'/nobackup/{user}/tmpdata/cossen/')

        if not os.path.isdir(self.target_location + 'train'):
            os.mkdir(self.target_location + 'train')
        if not os.path.isdir(self.target_location + 'test'):
            os.mkdir(self.target_location + 'test')

        copyfile(self.location + 'train_labels.csv', self.target_location + 'train_labels.csv')
        copyfile(self.location + 'test_labels.csv', self.target_location + 'test_labels.csv')

        self.num_train_files = len([name for name in os.listdir(self.location + 'train/') if name.endswith('tar')])

    def get_X_y(self, subsample=1.0, test=False):
        if test:
            labels = pd.read_csv(self.location + 'test_labels.csv')
            self._load_single(self.location + 'test/test.tar', self.target_location + 'test/')
            X_test = []
            y_test = np.asarray(labels['Location+MT'].apply(eval).values)
            for f in labels['Example#']:
                X_test.append(self._single_image_to_array(self.target_location + 'test/' + f))
            X_test = np.asarray(X_test)

        if subsample < 1:
            files = list(map(lambda a: self.location + 'train/' + str(a).zfill(3) + '.tar',
                             np.random.randint(0, self.num_train_files, size=int(subsample * self.num_train_files))))
        else:
            files = list(
                map(lambda a: self.location + 'train/' + str(a).zfill(3) + '.tar', np.arange(self.num_train_files)))

        for f in tqdm(files, desc='Transferring files'):
            self._load_single(f, self.target_location + 'train/')

        labels = pd.read_csv(self.location + 'train_labels.csv')

        to_filter = list(map(lambda f: f.split('/')[-1].split('.')[0], files))
        labels['Example#'] = labels['Example#'].apply(lambda a: a.split('.')[0].zfill(16))

        idx = np.asarray([labels['Example#'].str.endswith(ft).astype(int).values for ft in to_filter]).sum(axis=0)
        labels = labels[idx > 0]

        X = []
        y = np.asarray(labels['Location+MT'].apply(eval).values)
        for f in tqdm(labels['Example#'],desc='Loading images'):
            print(self.target_location + 'train/' + int(f)+'.png')
            X.append(self._single_image_to_array(self.target_location + 'train/' + int(f)+'.png'))
        X = np.asarray(X)

        if test:
            return X, X_test, y, y_test
        else:
            return X, y

    def _single_image_to_array(self, filename):
        image = Image.open(filename).convert('L')
        return np.array(image)

    def _load_single(self, filename, extract_to):
        f = tarfile.open(filename)
        f.extractall(extract_to)
        f.close()