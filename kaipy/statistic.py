# This file is part of kaipy.
# Copyright (C) 2016  Kai Szuttor
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

# Parts of the code were taken from https://github.com/statsmodels:
# Copyright (C) 2006, Jonathan E. Taylor
# All rights reserved.
# Copyright (c) 2006-2008 Scipy Developers.
# All rights reserved.
# Copyright (c) 2009-2013 Statsmodels Developers.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided
# with the distribution.
# 3. The name of the author may not be used to endorse or promote
# products derived from this software without specific prior
# written permission.
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np


def _bit_length_26(x): #pragma: no cover
    if x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        return len(bin(x)) - 2


def _next_regular(target): #pragma: no cover
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** _bit_length_26(quotient - 1)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


def autocorrelation(x, nlags=50):
    """
    Compute autocorrelation using FFT
    """
    nobs = len(x)
    x = np.squeeze(np.asarray(x))
    x0 = x - x.mean()
    n = _next_regular(2 * nobs + 1)
    Frf = np.fft.fft(x0, n=n)
    acf = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / nobs
    acf /= acf[0]
    acf = np.real(acf[:nlags + 1])
    return acf


def calc_error(data):
    """
    Error estimation for time series of simulation observables and take into
    account that these series are to some kind degree correlated (which
    enhances the estimated statistical error).
    """
    # calculate the normalized autocorrelation function of data
    acf = autocorrelation(data, nlags=len(data))
    # calculate the integrated correlation time tau_int
    # (Janke, Wolfhard. "Statistical analysis of simulations: Data correlations
    # and error estimation." Quantum Simulations of Complex Many-Body Systems:
    # From Theory to Algorithms 10 (2002): 423-445.)
    tau_int = 0.5
    k_max = len(acf)
    for k in range(1, k_max):
        tau_int += acf[k]
    # mean value of the time series
    data_mean = np.mean(data)
    # calculate the so called effective length of the time series N_eff
    if (tau_int > 0.5):
        N_eff = len(data) / (2.0 * tau_int)
        # finally the error is sqrt(var(data)/N_eff)
        stat_err = np.sqrt(np.var(data) / N_eff)
    else:
        stat_err = np.sqrt(np.var(data) / len(data))
    return data_mean, stat_err
