#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 23:45 2018/4/18

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   test
@IDE:       PyCharm
"""
import numpy as np
from spectra import OHSpectra, AddSpectra


oh_0 = OHSpectra(band='A-X', v_upper=0, v_lower=0)
oh_1 = OHSpectra(band='A-X', v_upper=1, v_lower=0)
oh_2 = OHSpectra(band='A-X', v_upper=1, v_lower=1)
oh = AddSpectra(spec0=oh_0, spec1=oh_1)
oh = AddSpectra(spec0=oh, spec1=oh_2)
oh.set_maxwell_distribution(Tvib=5000, Trot=2000)
oh.set_intensity()

wv = np.linspace(300, 320, num=3000)

