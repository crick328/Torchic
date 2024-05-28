from torchic.cuda_tensor import Tensor
from torchic.tools.helper import norm_and_flatten_images, labels_to_OHE
import tempfile
import numpy as np
# tutte robe che servono per parsare mnist og
import struct
import array
import functools
import operator

import os
os.environ['KAGGLE_CONFIG_DIR'] = r'your\dir'
from kaggle.api.kaggle_api_extended import KaggleApi



class DataLoader:
    api = KaggleApi()
    api.authenticate()
    directory = None

    def _preprocess_images(data: np.ndarray, device=None):
        return norm_and_flatten_images(data, device)
    
    def _to_ohe(data: np.ndarray, device=None):
        return labels_to_OHE(data, device)
    

class mnist(DataLoader):
    # codice riadattato da https://github.com/datapythonista/mnist
    directory = tempfile.gettempdir()
    DataLoader.api.dataset_download_files(
        dataset='hojjatk/mnist-dataset',    # name
        path=directory,         # dest. dir
        unzip=True)

    @classmethod
    def train_images(cls, device=None, preprocess=True):
        data = cls._parse_mnist(r"train-images-idx3-ubyte")
        if preprocess:
            return cls._preprocess_images(data, device)
        return data

    @classmethod
    def test_images(cls, device=None, preprocess=True):
        data = cls._parse_mnist(r"t10k-images-idx3-ubyte")
        if preprocess:
            return cls._preprocess_images(data, device)
        return data

    @classmethod
    def train_labels(cls, device=None, one_hot_encode=True):
        labels = cls._parse_mnist(r"train-labels-idx1-ubyte")
        if one_hot_encode:
            return cls._to_ohe(labels, device)
        return labels

    @classmethod
    def test_labels(cls, device=None, one_hot_encode=True):
        labels = cls._parse_mnist(r"t10k-labels-idx1-ubyte")
        if one_hot_encode:
            return cls._to_ohe(labels, device)
        return labels

    @classmethod
    def _parse_mnist(cls, filename):
        # doppio filename perche' dezzippa sia come filename e sia in una
        # cartella filename/filename (e python da errore quando prova a 
        # leggere la prima)
        trg_filename = os.path.join(mnist.directory, filename, filename)
        print(trg_filename)
        #fopen = gzip.open if os.path.splitext(filename)[1] == '.gz' else open

        with open(trg_filename, 'rb') as fd:
            return mnist._parse_idx(fd)

    @classmethod
    def _parse_idx(cls, fd):
        """Parse an IDX file, and return it as a numpy array.

        Parameters
        ----------
        fd : file
            File descriptor of the IDX file to parse

        endian : str
            Byte order of the IDX file. See [1] for available options

        Returns
        -------
        data : numpy.ndarray
            Numpy array with the dimensions and the data in the IDX file

        1. https://docs.python.org/3/library/struct.html
            #byte-order-size-and-alignment
        """
        DATA_TYPES = {0x08: 'B',  # unsigned byte
                    0x09: 'b',  # signed byte
                    0x0b: 'h',  # short (2 bytes)
                    0x0c: 'i',  # int (4 bytes)
                    0x0d: 'f',  # float (4 bytes)
                    0x0e: 'd'}  # double (8 bytes)

        header = fd.read(4)
        if len(header) != 4:
            raise ValueError('Invalid IDX file, '
                                'file empty or does not contain a full header.')

        zeros, data_type, num_dimensions = struct.unpack('>HBB', header)

        if zeros != 0:
            raise ValueError('Invalid IDX file, '
                                'file must start with two zero bytes. '
                                'Found 0x%02x' % zeros)

        try:
            data_type = DATA_TYPES[data_type]
        except KeyError:
            raise ValueError('Unknown data type '
                                '0x%02x in IDX file' % data_type)

        dimension_sizes = struct.unpack('>' + 'I' * num_dimensions,
                                        fd.read(4 * num_dimensions))

        data = array.array(data_type, fd.read())
        data.byteswap()  # looks like array.array reads data as little endian

        expected_items = functools.reduce(operator.mul, dimension_sizes)
        if len(data) != expected_items:
            raise ValueError('IDX file has wrong number of items. '
                                'Expected: %d. Found: %d' % (expected_items,
                                                            len(data)))

        return np.array(data).reshape(dimension_sizes)