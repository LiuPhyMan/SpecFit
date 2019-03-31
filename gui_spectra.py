#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13:20 2018/4/6

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   test
@IDE:       PyCharm
"""
import os
import re
import sys
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import time
from spectra import OHSpectra
from lmfit import Model
from PyQt5 import QtWidgets as QW
from PyQt5.QtGui import QIcon, QFont, QCursor
from PyQt5.QtCore import Qt, pyqtSignal
from BetterQWidgets import BetterQPushButton
from BasicFunc import tracer
from qtwidget import (SpectraPlot,
                      RangeQWidget,
                      ReadFileQWidget,
                      ParaQWidget,
                      GoodnessOfFit,
                      SpectraFunc,
                      ReportQWidget,
                      BandBranchesQTreeWidget,
                      NormalizedQGroupBox,
                      SizeInput,
                      FitArgsInput,
                      ExpLinesQTreeWidget)

_DEFAULT_TOOLBAR_FONT = QFont("Helvetica", 10)


class GUISpectra(QW.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.showMaximized()
        self.cenWidget = QW.QWidget()
        self.setWindowTitle('Spectra sim v1.0')
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.setWindowIcon(QIcon(dir_path + r'\gui_materials/matplotlib_large.png'))
        self.setCentralWidget(self.cenWidget)

        init_width, init_height = 12, 5
        self._spectra_plot = SpectraPlot(self.cenWidget, width=init_width, height=init_height)
        self._file_read = ReadFileQWidget()
        self._spectra_tree = SpectraFunc()
        self._wavelength_range = RangeQWidget()
        self._parameters_input = ParaQWidget()
        self._report = ReportQWidget()
        self._branch_tree = BandBranchesQTreeWidget()
        self._resize_input = SizeInput(init_width=init_width, init_height=init_height)
        self._fit_kws_input = FitArgsInput()
        self._sim_result = None
        self._goodness_of_fit = GoodnessOfFit()
        self._normalized_factor = 1
        self._normalized_groupbox = NormalizedQGroupBox()
        self._exp_data = dict(wavelength=None, intensity=None)
        self._output = QW.QTextEdit()
        self._output.setFont(QFont("Consolas", 11))
        self._output.setFixedWidth(300)
        self._set_button_layout()
        self._set_layout()
        self._set_menubar()
        self._set_toolbar()
        self._set_dockwidget()
        self._set_connect()

    def _set_dockwidget(self):
        _default_features = QW.QDockWidget.DockWidgetClosable | QW.QDockWidget.DockWidgetFloatable
        _list = ['Branch', 'Resize', "Fit_Args", "Output", "Bands"]
        _widgets_to_dock = [self._branch_tree,
                            self._resize_input,
                            self._fit_kws_input,
                            self._output,
                            self._spectra_tree]
        _dock_dict = dict()
        for _, _widget in zip(_list, _widgets_to_dock):
            _dock_dict[_] = QW.QDockWidget(_, self)
            _dock_dict[_].setWidget(_widget)
            _dock_dict[_].setFeatures(_default_features)
            _dock_dict[_].setVisible(False)
            _dock_dict[_].setFloating(True)
            _dock_dict[_].setCursor(QCursor(Qt.PointingHandCursor))
            _action = _dock_dict[_].toggleViewAction()
            _action.setChecked(False)
            _action.setFont(_DEFAULT_TOOLBAR_FONT)
            _action.setText(_)
            self.toolbar.addAction(_action)

    def _set_menubar(self):
        menubar = self.menuBar()
        menubar.setFont(QFont('Ubuntu', 14))

    def _set_toolbar(self):
        self.toolbar = self.addToolBar('toolbar')
        save_action = QW.QAction('Save', self)
        report_action = QW.QAction('Report', self)
        save_action.setFont(_DEFAULT_TOOLBAR_FONT)
        report_action.setFont(_DEFAULT_TOOLBAR_FONT)
        self.toolbar.addAction(save_action)
        self.toolbar.addAction(report_action)
        report_action.triggered.connect(self.show_report)
        self.toolbar.addSeparator()

    @staticmethod
    def _get_pushbutton(text, *, height=None):
        _ = BetterQPushButton(text)
        if height:
            _.setFixedHeight(height)
        return _

    def _set_button_layout(self):
        self.button_layout = QW.QVBoxLayout()
        sub_layout = QW.QGridLayout()
        self._plot_buttons = dict()
        self._plot_buttons['clear_sim'] = self._get_pushbutton("ClearSim", height=30)
        self._plot_buttons['clear_exp'] = self._get_pushbutton('ClearExp', height=30)
        self._plot_buttons['add_sim'] = self._get_pushbutton("AddSim", height=30)
        self._plot_buttons['auto_scale'] = self._get_pushbutton("&AutoScale", height=30)
        self._plot_buttons['fit'] = self._get_pushbutton('&Fit', height=60)
        self._plot_buttons['FitDistrbtn'] = self._get_pushbutton('FitDistrbtn', height=30)
        sub_layout.addWidget(self._plot_buttons['auto_scale'], 0, 0)
        sub_layout.addWidget(self._plot_buttons['fit'], 1, 0)
        sub_layout.addWidget(self._plot_buttons['clear_sim'], 2, 0)
        sub_layout.addWidget(self._plot_buttons['clear_exp'], 3, 0)
        sub_layout.addWidget(self._plot_buttons['FitDistrbtn'], 4, 0)
        self.button_layout.addLayout(sub_layout)
        self.button_layout.addStretch(1)

    def _set_layout(self):
        _layout = QW.QVBoxLayout()
        _layout.addWidget(self._spectra_plot)
        _layout.addWidget(self._file_read)

        sub_layout = QW.QHBoxLayout()
        left_layout = QW.QVBoxLayout()
        left_layout.addWidget(self._wavelength_range)
        # left_layout.addWidget(self._spectra_tree)
        left_layout.addStretch(1)
        sub_layout.addLayout(left_layout)
        sub_layout.addWidget(self._parameters_input)
        sub_layout.addWidget(self._normalized_groupbox)
        sub_layout.addLayout(self.button_layout)
        sub_layout.addWidget(self._goodness_of_fit)
        sub_layout.addStretch(1)
        _layout.addLayout(sub_layout)
        _layout.addStretch(1)
        self.cenWidget.setLayout(_layout)

    def _set_connect(self):
        def _file_read_callback():
            xdata = []
            ydata = []
            file_name = self._file_read._entry.text()
            if file_name == '':
                return
            with open(file_name, 'r') as f:
                for line in f:
                    if not re.fullmatch(r"\S+\s+\S+", line.strip()):
                        continue
                    num0_str, num1_str = re.fullmatch(r"(\S+)\s+(\S+)", line.strip()).groups()
                    try:
                        float(num0_str)
                    except:
                        continue
                    try:
                        float(num1_str)
                    except:
                        continue
                    xdata.append(float(num0_str))
                    ydata.append(float(num1_str))

            xdata = np.array(xdata)
            ydata = np.array(ydata)
            if self._normalized_groupbox.is_nomalized():
                _factor = ydata.max()
                ydata = ydata / _factor
                self._normalized_groupbox.set_value(_factor)
                self._normalized_factor = _factor

            self._spectra_plot.add_exp_line(xdata=xdata, ydata=ydata)
            self._exp_data = dict()
            self._exp_data['wavelength'] = xdata
            self._exp_data['intensity'] = ydata
            self._spectra_plot.auto_scale()

        def _parameters_input_callback():
            if self._exp_data['wavelength'] is not None:
                self.plot_sim()

        def _resize_plot(args):
            width = args[0]
            height = args[1]
            self._spectra_plot.setFixedWidth(width * 100)
            self._spectra_plot.setFixedHeight(height * 100)
            self._spectra_plot.canvas.setFixedWidth(width * 100 * 0.9)
            self._spectra_plot.canvas.setFixedHeight(height * 100 * 0.9)

        def _branch_tree_callback():
            self._spectra_plot.cls_line_intensity()
            self._spectra_plot.cls_texts()
            for _band, _v_upper, _v_lower, _branch in self._branch_tree.get_select_state():
                self.plot_line_intensity(band=_band, v_upper=_v_upper, v_lower=_v_lower,
                                         branch=_branch)

        def plot_mouse_move_callback(event):
            if event.inaxes:
                self.statusBar().showMessage('x={x:.2f}, y={y:.2f}'.format(x=event.xdata,
                                                                           y=event.ydata))
            else:
                self.statusBar().showMessage('Ready')

        #   file read, parameters input, buttons
        self._file_read.TextChangedSignal.connect(_file_read_callback)
        self._parameters_input.valueChanged.connect(_parameters_input_callback)
        self._plot_buttons['fit'].clicked.connect(self.sim_exp)
        self._plot_buttons['clear_exp'].clicked.connect(self._spectra_plot.cls_exp_lines)
        self._plot_buttons['clear_sim'].clicked.connect(self._spectra_plot.cls_sim_line)
        self._plot_buttons['auto_scale'].clicked.connect(self._spectra_plot.auto_scale)
        self._plot_buttons['FitDistrbtn'].clicked.connect(self.sim_exp_by_distribution)
        #   spectra plot
        self._spectra_plot.figure.canvas.mpl_connect('motion_notify_event',
                                                     plot_mouse_move_callback)
        #   resize, branch_tree AT dockwidgets.
        self._resize_input.valueChanged.connect(lambda: _resize_plot(self._resize_input.value()))
        self._branch_tree.stateChanged.connect(_branch_tree_callback)

    # ------------------------------------------------------------------------------------------- #
    def wave_intens_exp_in_range(self):
        r"""Returns wave_exp, intensity_exp in the range."""
        wv_range = self._wavelength_range.value()
        wave_exp = self._exp_data["wavelength"]
        intens_exp = self._exp_data["intensity"]
        _boolean_in_range = np.logical_and(wv_range[0] < wave_exp, wave_exp < wv_range[1])
        wave_exp_in_range = wave_exp[_boolean_in_range]
        intens_exp_in_range = intens_exp[_boolean_in_range]
        return wave_exp_in_range, intens_exp_in_range

    # ------------------------------------------------------------------------------------------- #
    def _evolve_spectra(self, _spc_func, wv_range, slit_func_name,
                        x, Tvib, Trot_cold, Trot_hot,
                        hot_ratio, fwhm_g, fwhm_l,
                        x_offset_x0, x_offset_k0, x_offset_k1, x_offset_k2, x_offset_k3,
                        y_offset_x0, y_offset_k0, y_offset_c0, y_offset_I0):
        # --------------------------------------------------------------------------------------- #
        # Trace the variation of the parameters.
        _str_0 = 'Trot : {Trot:6.0f} K, Tvib : {Tvib:6.0f} K\n'.format(Tvib=Tvib, Trot=Trot_cold)
        _str_1 = '  wG : {g:6.1f} pm\n  wL : {l:6.1f} pm\n'.format(g=fwhm_g * 1e3, l=fwhm_l * 1e3)
        print(_str_0 + _str_1)
        # --------------------------------------------------------------------------------------- #
        #   1. Set distribution.
        #       This part is associated with Tvib, Trot_cold, Trot_hot, hot_ratio.
        if self._parameters_input._temperature.para_form() == 'one_Trot':
            _spc_func.set_maxwell_distribution(Tvib=Tvib, Trot=Trot_cold)
        else:
            _spc_func.set_double_temperature_distribution(Tvib=Tvib,
                                                          Trot_cold=Trot_cold,
                                                          Trot_hot=Trot_hot,
                                                          hot_ratio=hot_ratio)
        # --------------------------------------------------------------------------------------- #
        #   2. Set intensity without profile function.
        _spc_func.set_intensity()
        # --------------------------------------------------------------------------------------- #
        #   3. Correct the wavelength and its range.
        #       This part is associated with the wavelength correct.
        x_correct_kwargs = dict(x0=x_offset_x0,
                                k0=x_offset_k0, k1=x_offset_k1, k2=x_offset_k2, k3=x_offset_k3)
        y_correct_kwargs = dict(x0=y_offset_x0,
                                k0=y_offset_k0, c0=y_offset_c0, I0=y_offset_I0)
        self.x_correct_func = self._parameters_input._x_offset.correct_func(**x_correct_kwargs)
        self.y_correct_func = self._parameters_input._y_offset.correct_func(**y_correct_kwargs)
        self.x_correct_func_reversed = self._parameters_input._x_offset.correct_func_reversed(
            **x_correct_kwargs)
        #   Calculate the absolute wavelength position to evolve the intensity and its range.
        wave_range_corrected = self.x_correct_func(wv_range)
        wavelength_corrected = self.x_correct_func(x)
        # --------------------------------------------------------------------------------------- #
        #   4. Evolve the spectra on the corrected wavelenth.
        _, intens = _spc_func.get_extended_wavelength(waveLength_exp=wavelength_corrected,
                                                      wavelength_range=wave_range_corrected,
                                                      slit_func=slit_func_name,
                                                      fwhm={'Gaussian': fwhm_g,
                                                            'Lorentzian': fwhm_l},
                                                      normalized=True)
        # --------------------------------------------------------------------------------------- #
        #   5. Correct the intensity.
        _corrected_intensity = self.y_correct_func(_, intens)
        return _corrected_intensity

    # ------------------------------------------------------------------------------------------- #
    def get_sim_data(self):
        r"""
        Get the synthetic spectra based on the parameters on the panel.
        """
        spc_func = self._spectra_tree.spectra_func
        wv_range = self._wavelength_range.value()
        slit_func = self._parameters_input._fwhm.para_form()
        intensity_simulated = self._evolve_spectra(spc_func, wv_range, slit_func,
                                                   self._exp_data['wavelength'],
                                                   *self._parameters_input.value())
        wave_exp_in_range, intens_exp_in_range = self.wave_intens_exp_in_range()
        self._goodness_of_fit.set_value(p_data=intensity_simulated, o_data=intens_exp_in_range)
        return wave_exp_in_range, intensity_simulated

    def plot_sim(self):
        self._spectra_plot.cls_sim_line()
        xdata, ydata = self.get_sim_data()
        self._spectra_plot.set_sim_line(xdata=xdata, ydata=ydata)

    # ------------------------------------------------------------------------------------------- #
    def _get_line_intensity(self, band, v_upper, v_lower, branch):
        paras_dict = self._parameters_input.value()
        Tvib, Trot_cold, Trot_hot, hot_ratio = paras_dict[:4]
        Trot_para_form = self._parameters_input._temperature.para_form()

        # set spectra function
        if band == 'OH(A-X)':
            _spc_func = OHSpectra(band='A-X', v_upper=v_upper, v_lower=v_lower)
        # set distribution
        if Trot_para_form == 'one_Trot':
            _spc_func.set_maxwell_distribution(Tvib=Tvib, Trot=Trot_cold)
        else:
            _spc_func.set_double_temperature_distribution(Tvib=Tvib,
                                                          Trot_cold=Trot_cold,
                                                          Trot_hot=Trot_hot,
                                                          hot_ratio=hot_ratio)
        # set intensity
        _spc_func.set_intensity()
        wv, intens = _spc_func.line_intensity(branch=branch)
        _normalized_factor = self._spectra_tree.spectra_func.normalized_factor
        return self.x_correct_func_reversed(wv), self.y_correct_func(wv,
                                                                     intens / _normalized_factor)

    def plot_line_intensity(self, *, band, v_upper, v_lower, branch):
        x, y = self._get_line_intensity(band=band, v_upper=v_upper, v_lower=v_lower, branch=branch)
        N_max = x.size
        if band == 'OH(A-X)':
            shown_index = np.arange(-1, N_max, 2)
            if branch in ('P2', 'R2'):
                x = x[1:]
                y = y[1:]
                shown_index[0] = 1
                shown_index = shown_index - 1
            else:
                shown_index[0] = 0
            shown_J = np.arange(x.size) + 1
        self._spectra_plot.plot_line_intensity(xdata=x, ydata=y,
                                               branch=branch, shown_index=shown_index,
                                               shown_J=shown_J)

    # ------------------------------------------------------------------------------------------- #
    def sim_exp(self):
        _spc_func = self._spectra_tree.spectra_func
        wv_range = self._wavelength_range.value()
        slit_func_name = self._parameters_input._fwhm.para_form()

        def fit_func(x, Tvib, Trot_cold, Trot_hot, hot_ratio, fwhm_g, fwhm_l,
                     x_offset_x0, x_offset_k0, x_offset_k1, x_offset_k2, x_offset_k3,
                     y_offset_x0, y_offset_k0, y_offset_c0, y_offset_I0):
            return self._evolve_spectra(_spc_func, wv_range, slit_func_name,
                                        x, Tvib, Trot_cold, Trot_hot, hot_ratio,
                                        fwhm_g, fwhm_l,
                                        x_offset_x0, x_offset_k0, x_offset_k1, x_offset_k2,
                                        x_offset_k3,
                                        y_offset_x0, y_offset_k0, y_offset_c0, y_offset_I0)

        # --------------------------------------------------------------------------------------- #
        #   Build model
        spectra_fit_model = Model(fit_func)
        # --------------------------------------------------------------------------------------- #
        #   Parameters
        #       init values
        #       range
        #       vary
        params = spectra_fit_model.make_params()
        init_value = self._parameters_input.value()
        range_dict = dict(Tvib=(300, 10000),
                          Trot_cold=(300, 10000),
                          Trot_hot=(300, 20000),
                          hot_ratio=(0, 1),
                          fwhm_g=(0, 1),
                          fwhm_l=(0, 1),
                          x_offset_x0=(0, 1000),
                          x_offset_k0=(-np.inf, np.inf),
                          x_offset_k1=(-np.inf, np.inf),
                          x_offset_k2=(-np.inf, np.inf),
                          x_offset_k3=(-np.inf, np.inf),
                          y_offset_x0=(-np.inf, np.inf),
                          y_offset_k0=(-np.inf, np.inf),
                          y_offset_c0=(-np.inf, np.inf),
                          y_offset_I0=(0, np.inf))
        for _i, _key in enumerate(('Tvib', 'Trot_cold', 'Trot_hot', 'hot_ratio',
                                   'fwhm_g', 'fwhm_l',
                                   'x_offset_x0', 'x_offset_k0', 'x_offset_k1', 'x_offset_k2',
                                   'x_offset_k3',
                                   'y_offset_x0', 'y_offset_k0', 'y_offset_c0', 'y_offset_I0')):
            params[_key].set(min=range_dict[_key][0], max=range_dict[_key][1])
            params[_key].set(vary=self._parameters_input.is_variable_state()[_i])
            params[_key].set(value=init_value[_i])

        # --------------------------------------------------------------------------------------- #
        wave_exp_in_range, intens_exp_in_range = self.wave_intens_exp_in_range()
        self._sim_result = spectra_fit_model.fit(intens_exp_in_range, params=params,
                                                 method='least_squares',
                                                 fit_kws=self._fit_kws_input.value(),
                                                 x=wave_exp_in_range)
        self.spectra_fit_model = spectra_fit_model
        #   Plot the simulated line on the spectra plot.
        self._spectra_plot.cls_sim_line()
        self._spectra_plot.set_sim_line(xdata=wave_exp_in_range, ydata=self._sim_result.best_fit)
        #   Output the simulated variables to the parameters panel.
        self._parameters_input.set_value(**self._sim_result.values)
        #   Show the goodness of fit.
        self._goodness_of_fit.set_value(p_data=self._sim_result.best_fit,
                                        o_data=self._sim_result.data)
        self._output.setText(self.simulated_result_str())
        #   Show report.
        self.show_report()

    def sim_exp_by_distribution(self):
        _spc_func = self._spectra_tree.spectra_func
        wv_range = self._wavelength_range.value()

        x_correct_kwargs = self._parameters_input._x_offset.value()
        y_correct_kwargs = self._parameters_input._y_offset.value()

        x_correct_func = self._parameters_input._x_offset.correct_func(**x_correct_kwargs)
        y_correct_func = self._parameters_input._y_offset.correct_func(**y_correct_kwargs)

        slit_func_name = self._parameters_input._fwhm.para_form()
        fwhm_g = self._parameters_input._fwhm.value()['fwhm_g']
        fwhm_l = self._parameters_input._fwhm.value()['fwhm_l']

        wv_exp_in_range, intens_exp_in_range = self.wave_intens_exp_in_range()
        wave_range_corrected = x_correct_func(wv_range)
        wavelength_corrected = x_correct_func(wv_exp_in_range)
        # _spc_func = _spc_func.narrow_range(_range=wave_range_corrected)
        _spc_func.set_maxwell_distribution(Tvib=4000, Trot=3000)
        distribution_guess = np.hstack((_spc_func.get_level_params(_spc_func.distribution)[0],
                                        _spc_func.get_level_params(_spc_func.distribution)[1]))

        @tracer
        def fit_func(x, *_distribution):
            #   Now only support OH(A-X, 0-0) band.
            _distri_array = np.array(_distribution)
            _spc_func.set_distribution(F1_distri=_distri_array[:42],
                                       F2_distri=_distri_array[42:])
            _spc_func.set_intensity()
            _, in_sim = _spc_func.get_extended_wavelength(waveLength_exp=wavelength_corrected,
                                                          wavelength_range=wave_range_corrected,
                                                          slit_func=slit_func_name,
                                                          fwhm=dict(Gaussian=fwhm_g,
                                                                    Lorentzian=fwhm_l),
                                                          normalized=True)
            _corrected_in_sim = y_correct_func(wv_exp_in_range, in_sim)
            return _corrected_in_sim

        # --------------------------------------------------------------------------------------- #
        plt.figure()
        plt.plot(wv_exp_in_range, intens_exp_in_range)
        _temp_in_sim = fit_func(wv_exp_in_range, *distribution_guess)
        plt.plot(wv_exp_in_range, _temp_in_sim)
        # --------------------------------------------------------------------------------------- #

        distribution_fitted, pcov = curve_fit(fit_func,
                                              wv_exp_in_range,
                                              intens_exp_in_range,
                                              bounds=(0, np.inf),
                                              p0=distribution_guess)

        Fev_upper = _spc_func.get_level_params(_spc_func.Fev_upper)[0]
        gJ_upper = _spc_func.get_level_params(_spc_func.gJ_upper)[0]
        # --------------------------------------------------------------------------------------- #
        # plt.figure()
        # plt.plot(wv_exp_in_range, intens_exp_in_range)
        _temp_in_sim = fit_func(wv_exp_in_range, *distribution_fitted)
        plt.plot(wv_exp_in_range, _temp_in_sim, marker='.')
        # --------------------------------------------------------------------------------------- #
        plt.figure()
        plt.semilogy(Fev_upper[:25], (distribution_guess[:42] / gJ_upper)[:25])
        plt.semilogy(Fev_upper[:25], (distribution_fitted[:42] / gJ_upper)[:25])
        # plt.plot(_spc_func.Fev_upper, distribution_fitted/_spc_func.gJ_upper, '.')
        print(distribution_fitted)

    # ------------------------------------------------------------------------------------------- #
    def simulated_result_str(self):
        def get_print_str(param, _format):
            if param.name.startswith('fwhm'):
                if param.vary:
                    return '\n{v:{frmt}} +/- {err:{frmt}} pm'.format(v=param.value * 1e3,
                                                                     err=param.stderr * 1e3,
                                                                     frmt=_format)
                else:
                    return '\n{v:{frmt}} [fixed]'.format(value=param.value,
                                                         frmt=_format)
            if param.vary:
                return '\n{v:{frmt}} +/- {err:{frmt}}'.format(v=param.value, err=param.stderr,
                                                              frmt=_format)
            else:
                return '\n{v:{frmt}} [fixed]'.format(value=param.value, frmt=_format)

        _str = r"""Simulation is {a}
        Method              : {b}
        ndata points        : {c}
        variables           : {d}
        function evals      : {e}
        reduced chi_square  : {f:.2e}
        R2                  : {g:.4f}
        Tvib
        Trot_cold
        fwhm_g
        fwhm_l
        """.format(a="Success" if self._sim_result.success else "Failed",
                   b=self._sim_result.method,
                   c=self._sim_result.ndata,
                   d=self._sim_result.nvarys,
                   e=self._sim_result.nfev,
                   f=self._sim_result.redchi,
                   g=self._goodness_of_fit._r2)
        _str += get_print_str(self._sim_result.params['Tvib'], '<8.2f')
        _str += get_print_str(self._sim_result.params['Trot_cold'], '<8.2f')
        _str += get_print_str(self._sim_result.params['fwhm_g'], '<8.3f')
        _str += get_print_str(self._sim_result.params['fwhm_l'], '<8.3f')
        _str += "\nftol : {f:<8.2e}".format(f=self._fit_kws_input.value()['ftol'])
        _str += "\nxtol : {f:<8.2e}".format(f=self._fit_kws_input.value()['xtol'])
        _str = re.sub(r"^\s+", r"\n", _str)
        _str = re.sub(r"\n\s+", r"\n", _str)
        return _str

    def show_report(self):
        _str = '' if self._sim_result is None else self._sim_result.fit_report()
        msg = QW.QMessageBox()
        msg.setIcon(QW.QMessageBox.Information)
        current_time = time.strftime('%Y/%m/%d %H:%M:%S')
        msg.setText(current_time)
        msg.setInformativeText(_str)
        msg.setFont(QFont('Consolas', 12))
        msg.setWindowTitle('Simulation Report')
        msg.setStandardButtons(QW.QMessageBox.Close)
        msg.exec_()


# ----------------------------------------------------------------------------------------------- #
class Temp(QW.QMainWindow):

    def __init__(self, parent=None):
        # super().__init__(parent)
        # self.resize(800, 600)
        # self.cenWidget = QW.QWidget()
        # self.setCentralWidget(self.cenWidget)
        #
        # btn = QW.QPushButton()
        # btn.clicked.connect(self.show_dialog)
        # btn1 = QW.QPushButton()
        # btn1.clicked.connect(self.show_dialog)
        #
        # layout = QW.QVBoxLayout()
        # layout.addWidget(btn)
        # layout.addWidget(btn1)
        #
        # self.cenWidget.setLayout(layout)

        super().__init__(parent)
        self.resize(800, 600)
        self.showMaximized()
        # self.showMinimized()
        # avGeom = QW.QDesktopWidget().availableGeometry()
        # self.setGeometry(avGeom)
        self.cenWidget = QW.QWidget()
        self.setWindowTitle('Spectra sim v1.0')
        self.setWindowIcon(QIcon('gui_materials/matplotlib_large.png'))
        self.setCentralWidget(self.cenWidget)
        exp_line = ExpLinesQTreeWidget()
        _layout = QW.QVBoxLayout()
        _layout.addWidget(exp_line)
        self.cenWidget.setLayout(_layout)


# ----------------------------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not QW.QApplication.instance():
        app = QW.QApplication(sys.argv)
    else:
        app = QW.QApplication.instance()
    app.setStyle(QW.QStyleFactory.create("WindowsVista"))
    window = GUISpectra()
    # window = Temp()
    window.show()
    # app.exec_()
    app.aboutToQuit.connect(app.deleteLater)
    # add self_branch tag