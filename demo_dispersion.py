#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 22:43 2018/5/29

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   test
@IDE:       PyCharm
"""
import math
import numpy as np
from scipy import special

def L(x,  f):
    gama = f / 2
    return gama / math.pi / (x ** 2 + gama ** 2)


def G(x, f):
    sigma = f / 2 / math.sqrt(2 * math.log(2))
    return np.exp(-x ** 2 / 2 / sigma ** 2) / sigma / math.sqrt(2 * math.pi)


def V_p(x, fL, fG):
    f = fG ** 5 + 2.69269 * fG ** 4 * fL + 2.42843 * fG ** 3 * fL ** 2 + \
        4.47163 * fG ** 2 * fL ** 3 + 0.07842 * fG * fL ** 4 + fL ** 5
    f = f ** (1 / 5)
    a = 1.36603 * (fL / f) - 0.47719 * (fL / f) ** 2 + 0.11116 * (fL / f) ** 3
    result = a * L(x, f) + (1 - a) * G(x, f)
    return result
#
#
#
#
# x = np.linspace(-10, 10, num=10000)
# L_line = L(x, 1)
# G_line = G(x, 1)
# V_p_line = V_p(x, 1, 1)
# V_line = V(x, 1, 1)
#
# from matplotlib import pyplot as plt
#
# # plt.plot(x, L_line)
# # plt.plot(x, G_line)
# plt.plot(x, V_line)
# plt.plot(x, V_p_line)
# plt.show()
