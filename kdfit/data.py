#  Copyright 2021 by Benjamin J. Land (a.k.a. BenLand100)
#
#  This file is part of kdfit.
#
#  kdfit is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  kdfit is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with kdfit.  If not, see <https://www.gnu.org/licenses/>.

from .calculate import Calculation

import numpy as np
import os
try:
    import uproot4
except Exception:
    pass
try:
    import h5py
except Exception:
    pass

class DataLoader(Calculation):
    '''
    This is a generic data input Calculation. Create subclasses of Calculation
    to load data of other formats. Subclasses should load data when called.
    '''
    
    def __init__(self, name):
        super().__init__(name, [], constant=True)
        
    def calculate(self, inputs, verbose=False):
        return self


class HDF5Data(DataLoader):
    '''
    Assumes each dimension of an Observable is stored as a dataset in an HDF5
    file. The datasets should be one dimensional and indexed by event number.
    All should be the same shape. Will chain together multiple files into one
    larger dataset.
    '''

    def __init__(self, name, filenames, datasets, max_events=None):
        super().__init__(name)
        self.filenames = filenames
        self.datasets = datasets
        self.max_events = max_events
    
    def __call__(self):
        print('Loading:', ', '.join(self.filenames))
        data = [[] for ds in self.datasets]
        for fname in self.filenames:
            with h5py.File(fname, 'r') as hf:
                total = len(data[0])
                for j, ds in enumerate(self.datasets):
                    if self.max_events is not None:
                        to_read = self.max_events - total
                        ds = hf[ds]
                        if total > ds.shape[0]:
                            data[j].extend(ds[:])
                        else:
                            data[j].extend(ds[:to_read])
                    else:
                        data[j].extend(hf[ds][:])
            if self.max_events is not None and len(data[0]) >= self.max_events:
                break
        if self.max_events is not None:
            return np.asarray(data)[:, :self.max_events].T
        else:
            return np.asarray(data).T


class UniformHDF5Data(DataLoader):
    '''
    Same as HDF5Loader but will load data uniformly from the supplied
    filenames.
    '''

    def __init__(self, name, filenames, datasets, max_events=None, fraction=None):
        self.rng = np.random.default_rng()
        super().__init__(name)
        self.filenames = filenames
        self.datasets = datasets
        self.max_events = max_events
        self.fraction = fraction
        if self.fraction is not None and self.max_events is not None:
            raise ValueError('Cannot specify both fraction and max_events')

    def __call__(self):
        print('Loading:', ', '.join(self.filenames))
        data = [[] for ds in self.datasets]
        vsources = {dset: [] for dset in self.datasets}
        file_lengths = []
        for fname in self.filenames:
            with h5py.File(fname, 'r') as f:
                file_lengths.append(len(f[self.datasets[0]]))
        total_events = np.sum(file_lengths)
        file_ends = np.concat([[0],np.cumsum(file_lengths) - 1])  # Subtract one for 0-indexing
        indices = np.arange(0, total_events)
        if (self.max_events is None and self.fraction is None) or\
            (self.max_events is not None and self.max_events > len(indices)):
            select_indices = indices
        else:
            num_to_load = int(self.fraction * total_events) if self.fraction is not None else self.max_events
            select_indices = self.rng.choice(indices, num_to_load,
                                             shuffle=False, replace=False)
            select_indices = np.sort(select_indices)
        for fi, fname in enumerate(self.filenames):
            with h5py.File(fname, 'r') as f:
                load_ids = select_indices[(select_indices >= file_ends[fi]) & (select_indices < file_ends[fi+1])]
                load_ids = load_ids - file_ends[fi]
                for j, dset in enumerate(self.datasets):
                    data[j].extend(f[dset][()][load_ids])
        return np.asarray(data).T


class BinnedHDF5Data(DataLoader):
    '''
    Assumes data is pre-binned store in a dataset named 'binned'
    '''

    def __init__(self, name, filename, key=None):
        super().__init__(name)
        self.filename = filename
        self.key = (key if key is not None else 'binned')
    
    def __call__(self):
        print(self.filename)
        with h5py.File(self.filename, 'r') as hf:
            return hf[self.key][:]
        
class NPYData(DataLoader):
    '''
    Assumes each dimension of an Observable is stored as a dataset in an HDF5
    file. The datasets should be one dimensional and indexed by event number.
    All should be the same shape. Will chain together multiple files into one
    larger dataset.
    '''

    def __init__(self, name, filenames, indexes, ordering='ij'):
        super().__init__(name)
        self.filenames = filenames
        self.indexes = np.asarray(indexes, dtype=np.int32)
        self.ordering = ordering
        
    def __call__(self):
        print('Loading:', ', '.join(self.filenames))
        x_nij = []
        for fname in self.filenames:
            events = np.load(fname)
            if self.ordering == 'ij':
                x_ = events[:, self.indexes]
                x_nij.append(x_)
            elif self.ordering == 'ji':
                x_ = events[self.indexes, :]
                x_nij.append(x_.T)
            else:
                raise Exception('Unknown ordering')
        return np.concatenate(x_nij)
                    
class SNOPlusNTuple(DataLoader):

    def __init__(self, name, filenames, branches, max_events=None):
        super().__init__(name)
        self.filenames = filenames
        self.branches = branches
        self.max_events = max_events
        
    def __call__(self):
        print('Loading:', ', '.join(self.filenames))
        x_nij = []
        events = 0
        for fname in self.filenames:
            try:
                with uproot4.open(fname) as froot:
                    x_ji = np.asarray([froot['output'][branch].array() for branch in self.branches])
                    x_nij.append(x_ji.T)
                    events += x_ji.shape[1]
                if self.max_events is not None and events > self.max_events:
                    break
            except Exception:
                print('Couldn\'t read', fname)
        print('Loaded', events, 'events')
        return np.concatenate(x_nij)
