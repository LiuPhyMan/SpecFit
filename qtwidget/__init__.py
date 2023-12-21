#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 7:39 2018/3/30

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
from .widgets import *
from .spectra_widget import *

__all__ = [s for s in dir() if not s.startswith('_')]

# from numpy.testing import Tester

# test = Tester().test
