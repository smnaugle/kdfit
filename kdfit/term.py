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

import numpy as np
try:
    import cupy as cp
except:
    cp = np # Use numpy to emulate cupy on CPU
import itertools as it
from .calculate import Calculation
from .signal import Signal
        
class Sum(Calculation):
    '''
    A Calculation that adds a series of input Calculations
    '''
    def __init__(self, name, *inputs):
        super().__init__(name, inputs)
        self.name = name
    def calculate(self, inputs, verbose=False):
        return np.sum(inputs)
    
class UnbinnedNegativeLogLikelihoodFunction(Calculation):
    '''
    Calculates the negative log likelihood of data x_kj with a PDF given by
    the sum of the PDFs from some scaled signals. The signal scales, or number 
    of events for that signal, are the inputs to this calculation. Also 
    calculates the Poisson likelihood of observing k events given a mean event 
    rate that is the sum of the signal scales. The final result is the negative
    logarithm of the product of the data likelihood and Poisson likelihood that
    omits any terms that are constant as a function of input scales.
    '''
    def __init__(self, name, signals, observables):
        self.signals = signals
        self.observables = observables
        n_evs = [s.nev_param for s in signals]
        super().__init__(name,n_evs+signals+[observables])

    def calculate(self, inputs, verbose=False):
        n_evs = inputs[:len(self.signals)]
        signal_systs = inputs[len(self.signals):-1]
        x_kj = cp.ascontiguousarray(cp.asarray(inputs[-1]))
        if verbose:
            print('Evaluate:',', '.join(['%0.3f'%s for s in n_evs]))
        if cp == np: # No GPU acceleration
            pdf_sk = np.asarray([n*s.eval_pdf_multi(x_kj,systs=systs) for n,s,systs in zip(n_evs,self.signals,signal_systs)])
            res = np.sum(n_evs) - np.sum(np.log(np.sum(pdf_sk,axis=0)))
        else:
            pdf_sk = cp.asarray([n*s.eval_pdf_multi(x_kj,systs=systs,get=False) for n,s,systs in zip(n_evs,self.signals,signal_systs)])
            res = np.sum(n_evs) - cp.sum(cp.log(cp.sum(pdf_sk,axis=0))).get()
        if verbose:
            print('NLL:',res)
        return res

class BinnedNegativeLogLikelihoodFunction(Calculation):
    '''
    Calculates the negative log likelihood of binned data x_kj with a PDF given by
    the sum of the PDFs from some scaled signals. The signal scales, or number 
    of events for that signal, are the inputs to this calculation. Also 
    calculates the Poisson likelihood of observing k events given a mean event 
    rate that is the sum of the signal scales. The final result is the negative
    logarithm of the product of the data likelihood and Poisson likelihood that
    omits any terms that are constant as a function of input scales.
    '''
    def __init__(self, name, signals, observables, binning=21):
        self.binning = binning
        self.signals = signals
        self.observables = observables
        n_evs = [s.nev_param for s in signals]
        super().__init__(name,n_evs+signals+[observables])
        self.last_x_kj = None
    
    def calculate(self, inputs, verbose=False):
        n_evs = inputs[:len(self.signals)]
        signal_systs = inputs[len(self.signals):-1]
        x_kj = inputs[-1]
        if x_kj is not self.last_x_kj:
            if type(self.binning) == int:
                self.bin_edges = [cp.linspace(observables.lows[j],observables.highs[j],self.binning) for j in range(x_kj.shape[1])]
            else:
                self.bin_edges = [cp.linspace(observables.lows[j],observables.highs[j],bins) for j,bins in enumerate(self.binning)]
            self.bin_edges = cp.ascontiguousarray(cp.asarray(self.bin_edges))
            self.a_kj = cp.ascontiguousarray(cp.asarray([cp.asarray(x) for x in it.product(*self.bin_edges[:, :-1])]))
            self.b_kj = cp.ascontiguousarray(cp.asarray([cp.asarray(x) for x in it.product(*self.bin_edges[:,1:  ])]))
            self.bin_vol = cp.ascontiguousarray(cp.prod(self.b_kj-self.a_kj,axis=1))
            if np == cp:
                self.counts,_ = np.histogramdd(x_kj,bins=self.bin_edges)
            else:
                counts,_ = np.histogramdd(x_kj,bins=self.bin_edges.get())
                self.counts = cp.asarray(counts)
            self.last_x_kj = x_kj
        if verbose:
            print('Evaluate:',', '.join(['%0.3f'%s for s in n_evs]))
        signal_norms = [s.int_pdf(self.bin_edges[:,0],self.bin_edges[:,-1],systs=systs) for s,systs in zip(self.signals,signal_systs)]
        
        if cp == np: # No GPU acceleration
            expected = cp.sum([n*s.int_pdf_multi(self.a_kj,self.b_kj,systs=systs)/self.bin_vol/norm for n,s,systs,norm in zip(n_evs,self.signals,signal_systs,signal_norms)],axis=0)
        else:
            n_evs = cp.asarray(n_evs,cp.float64)
            signal_norms = cp.asarray(signal_norms,cp.float64)
            expected = [ n*s.int_pdf_multi(self.a_kj,self.b_kj,systs=systs,get=False)/self.bin_vol/norm
                         for n,s,systs,norm in zip(n_evs,self.signals,signal_systs,signal_norms) ]
            expected = cp.sum(cp.asarray(expected),axis=0)
        expected = expected.reshape(self.counts.shape)
        res = cp.sum(n_evs) - cp.sum(self.counts*cp.log(expected))
        if verbose:
            print('NLL:',res)
        return res if np == cp else res.get()
