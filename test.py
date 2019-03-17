#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  21:44 2019/3/17

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   SpecFit
@IDE:       PyCharm
"""

from spectra import N2Spectra
for i in range(5):
    N2 = N2Spectra(band='C-B', v_upper=i, v_lower=0)
    print(N2.Ge_upper[0,0])
