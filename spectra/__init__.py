#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18:01 2018/4/15

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   test
@IDE:       PyCharm
"""
from __future__ import division, print_function, absolute_import
from .spectra import *
# from . import voigt
from . import voigt

__all__ = [s for s in dir() if not s.startswith('_')]

from numpy.testing import Tester

test = Tester().test
