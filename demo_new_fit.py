#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  16:30 2019/3/11

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   SpecFit
@IDE:       PyCharm
"""
import numpy as np
from spectra import OHSpectra
from matplotlib import pyplot as plt

OH = OHSpectra(band='A-X', v_upper=0, v_lower=0)
OH.ravel_coefficients()

wv_exp = np.linspace(309.5, 310.5, num=50000)
# ----------------------------------------------------------------------------------------------- #
# OH.set_maxwell_distribution(Tvib=4000, Trot=2000)
_distribution = np.ones_like(OH.wave_length)
# _distribution[np.logical_and(309.0<OH.wave_length, OH.wave_length<309.2)] = 0
OH.set_distribution(distribution=_distribution)
# ----------------------------------------------------------------------------------------------- #
OH.set_intensity()
_, in_sim = OH.get_extended_wavelength(waveLength_exp=wv_exp,
                                       fwhm=dict(Gaussian=0.0001, Lorentzian=0.0002),
                                       slit_func='Voigt')
# ----------------------------------------------------------------------------------------------- #
plt.figure()
plt.plot(wv_exp, in_sim)
