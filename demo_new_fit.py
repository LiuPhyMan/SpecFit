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
from scipy.optimize import curve_fit
from spectra import OHSpectra
from matplotlib import pyplot as plt

OH = OHSpectra(band='A-X', v_upper=0, v_lower=0)
# OH.ravel_coefficients()

wv_exp = np.linspace(305.5, 319.5, num=500)
# OH = OH.narrow_range(_range=(wv_exp[0], wv_exp[-1]))
# ----------------------------------------------------------------------------------------------- #
OH.set_maxwell_distribution(Tvib=4000, Trot=2000)
# _distribution = np.ones_like(OH.wave_length)
# _distribution = np.abs(np.random.randn(OH.wave_length.size))
# _distribution[np.logical_and(309.0<OH.wave_length, OH.wave_length<309.2)] = 0
# OH.set_distribution(distribution=_distribution)
# ----------------------------------------------------------------------------------------------- #
OH.set_intensity()
_, in_sim_to_fit = OH.get_extended_wavelength(waveLength_exp=wv_exp,
                                              fwhm=dict(Gaussian=0.01, Lorentzian=0.02),
                                              slit_func='Voigt')

plt.figure()
plt.plot(wv_exp, in_sim_to_fit)

# ----------------------------------------------------------------------------------------------- #
# def func_fit(x, *distribution):
#     OH.set_distribution(distribution=np.array(distribution))
#     OH.set_intensity()
#     _, in_sim = OH.get_extended_wavelength(waveLength_exp=x,
#                                            fwhm=dict(Gaussian=0.01, Lorentzian=0.02),
#                                            slit_func='Voigt')
#     return in_sim
#
#
# distr_guess = np.ones_like(OH.wave_length)
# popt, pcov = curve_fit(func_fit, wv_exp, in_sim_to_fit, p0=distr_guess)
# perr = np.sqrt(np.diag(pcov))
#
# plt.errorbar(OH.Fev_upper, popt / OH.gJ_upper, yerr=perr * 1e10, fmt='o')
