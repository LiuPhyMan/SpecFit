#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on  10:04 2019/4/11

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   SpecFit
@IDE:       PyCharm
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
from BasicFunc import tracer

n_vars = 80
# from 1 to 10

init_vars = np.linspace(1, 10, num=n_vars)

x = np.linspace(0, 100, num=800)

@tracer
def func(x, *vars):
    return np.sin(x[np.newaxis].transpose() * np.array(vars)).sum(axis=1)


y_init = func(x, *init_vars)

# ----------------------------------------------------------------------------------------------- #
# def minimize_func(vars):
#     y_sim = func(x, vars)
#     return np.sum((y_sim-y_init)**2)

# res = minimize(minimize_func, vars_guess)
# print(res)
# ----------------------------------------------------------------------------------------------- #
vars_guess = init_vars + 1
popt, pcov = curve_fit(func, x, y_init, vars_guess)

