# This file is part of kaipy.
# Copyright (C) 2017  Kai Szuttor
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import math


def autocorrelation(data, normalized=True):
    """
    Compute autocorrelation using FFT
    """
    x = np.copy(data)
    nobs = len(x)
    x -= x.mean()
    n = 2**int(math.log(nobs, 2))
    x = x[:n]
    Frf = np.fft.fft(x)
    acf = np.fft.ifft(Frf * np.conjugate(Frf))/x.shape[0]
    if normalized:
        acf /= acf[0]
    acf = np.real(acf)
    # only return half of the ACF 
    # (see 4.3.1 "Kreuzkorrelationsfunktion" 
    # of https://github.com/arnolda/padc)
    return acf[:int(x.shape[0]/2)]


def calc_error(data):
    """
    Error estimation for time series of simulation observables and take into
    account that these series are correlated (which
    enhances the estimated statistical error).
    """
    # calculate the normalized autocorrelation function of data
    acf = autocorrelation(data)
    # calculate the integrated correlation time tau_int
    # (Janke, Wolfhard. "Statistical analysis of simulations: Data correlations
    # and error estimation." Quantum Simulations of Complex Many-Body Systems:
    # From Theory to Algorithms 10 (2002): 423-445.)
    tau_int = 0.5
    for i in range(len(acf)):
        tau_int += acf[i]
        if i >= 6 * tau_int:
            break
    # mean value of the time series
    data_mean = np.mean(data)
    # calculate the so called effective length of the time series N_eff
    if tau_int > 0.5:
        N_eff = len(data) / (2.0 * tau_int)
        # finally the error is sqrt(var(data)/N_eff)
        stat_err = np.sqrt(np.var(data) / N_eff)
    else:
        stat_err = np.sqrt(np.var(data) / len(data))
    return data_mean, stat_err
