#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15:43 2018/7/26

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   SpecFit
@IDE:       PyCharm
"""
import numpy as np
from matplotlib import pyplot as plt
from spectra import voigt_pseudo
from lmfit import Model
import re


def voigt(x, A, x0, y0, fG, fL):
    return voigt_pseudo(x - x0, fG, fL) * A + y0


def read_spec_asc(file_path):
    x = []
    y = []
    data_match = re.compile(r"[.\d]+\s+[\d.]+")
    with open(file_path) as f:
        for _line in f:
            if data_match.fullmatch(_line.strip()):
                x.append(float(_line.strip().split()[0]))
                y.append(float(_line.strip().split()[1]))
    return np.array(x), np.array(y)


vmodel = Model(voigt)
params = vmodel.make_params()
# x = np.linspace(-1, 1, num=300)
# y = voigt(x, 3.0, 0, 1, 0.1, 0.1)
for _i in range(17):
    x, y = read_spec_asc(r'D:\Desktop\{i}_um.asc'.format(i=_i * 10))
    y = y[np.logical_and(x < 667, 650 < x)]
    x = x[np.logical_and(x < 667, 650 < x)]

    params['A'].set(value=y.max() - y.min(), min=0, max=np.inf)
    params['x0'].set(value=656)
    params['y0'].set(value=y.min(), min=-np.inf, max=np.inf)
    params['fG'].set(value=0.03, min=0, max=1)
    params['fL'].set(value=0.03, min=0, max=1)
    out = vmodel.fit(y, params,
                     fit_kws=dict(ftol=1e-12, xtol=1e-12), x=x)
    # print(out.fit_report())
    # print(out.success)
    # print(out.redchi)
    out.plot()
    _str = '{g:.6f} {g_err:.6f} {l:.6f} {l_err:.6f} {A:.6f} {resi:.6f} {ratio:.6f}'
    print(_str.format(g=out.params['fG'].value,
                      g_err=out.params['fG'].stderr,
                      l=out.params['fL'].value,
                      l_err=out.params['fL'].stderr,
                      A=out.params['A'].value,
                      resi=out.residual.max(),
                      ratio=out.residual.max()/out.params['A'].value))
