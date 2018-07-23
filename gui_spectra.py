#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 13:20 2018/4/6

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   test
@IDE:       PyCharm
"""
import re
import sys
import numpy as np
import matplotlib
import copy
import time
from spectra import OHSpectra
from functools import partial
from lmfit import Model
from PyQt5 import QtWidgets as QW
from PyQt5.QtGui import QIcon, QCursor, QFont, QClipboard
from PyQt5.QtCore import Qt
from qtwidget import (SpectraPlot,
                      RangeQWidget,
                      ReadFileQWidget,
                      ParaQWidget,
                      GoodnessOfFit,
                      SpectraFunc,
                      ReportQWidget,
                      BetterButton,
                      BandBranchesQTreeWidget,
                      NormalizedQGroupBox,
                      SizeInput,
                      ExpLinesQTreeWidget)


class GUISpectra(QW.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.resize(800, 600)
        # self.showMaximized()
        # self.showMinimized()
        # avGeom = QW.QDesktopWidget().availableGeometry()
        # self.setGeometry(avGeom)
        self.cenWidget = QW.QWidget()
        self.setWindowTitle('Spectra sim v1.0')
        self.setWindowIcon(QIcon('gui_materials/matplotlib_large.png'))
        self.setCentralWidget(self.cenWidget)

        self._spectra_plot = SpectraPlot(self.cenWidget, width=17, height=6)
        self._file_read = ReadFileQWidget()
        self._spectra_tree = SpectraFunc()
        self._wavelength_range = RangeQWidget()
        self._parameters_input = ParaQWidget()
        self._report = ReportQWidget()
        self._branch_tree = BandBranchesQTreeWidget()
        self._resize_input = SizeInput(init_height=6, init_width=17)
        self._sim_result = None
        self._goodness_of_fit = GoodnessOfFit()
        self._normalized_factor = 1
        self._normalized_groupbox = NormalizedQGroupBox()
        self._exp_data = dict(wavelength=None, intensity=None)
        self.set_button_layout()
        self._set_layout()
        self._set_toolbar()
        self._set_connect()
        self._set_menubar()
        self._set_dockwidget()

    def _set_dockwidget(self):
        _default_features = QW.QDockWidget.DockWidgetClosable | QW.QDockWidget.DockWidgetFloatable
        _list = ['Branch', 'Resize']
        _dock_dict = dict()
        for _ in _list:
            _dock_dict[_] = QW.QDockWidget(_, self)
            if _ == 'Branch':
                _dock_dict[_].setWidget(self._branch_tree)
            else:
                _dock_dict[_].setWidget(self._resize_input)
            _dock_dict[_].setFeatures(_default_features)
            _dock_dict[_].setVisible(False)
            _dock_dict[_].setFloating(True)
            _action = _dock_dict[_].toggleViewAction()
            _action.setCheckable(True)
            _action.setChecked(False)
            _action.setFont(QFont('Ubuntu', 14))
            _action.setText(_)
            self.toolbar.addAction(_action)

    def _set_menubar(self):
        menubar = self.menuBar()
        menubar.setFont(QFont('Ubuntu', 14))

    def _set_toolbar(self):
        self.toolbar = self.addToolBar('toolbar')

        save_action = QW.QAction('Save', self)
        save_action.setFont(QFont('Ubuntu', 15))
        report_action = QW.QAction('Report', self)
        report_action.setFont(QFont('Ubuntu', 15))
        resize_action = QW.QAction('resize', self)
        resize_action.setFont(QFont('Ubuntu', 15))

        self.toolbar.addAction(save_action)
        self.toolbar.addAction(report_action)
        report_action.triggered.connect(self.show_report)

    def _set_layout(self):
        _layout = QW.QVBoxLayout()
        _layout.addWidget(self._spectra_plot)
        _layout.addWidget(self._file_read)

        sub_layout = QW.QHBoxLayout()
        left_layout = QW.QVBoxLayout()
        left_layout.addWidget(self._wavelength_range)
        left_layout.addWidget(self._spectra_tree)
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
            self._wavelength_range.set_value(_min=xdata.min(),
                                             _max=xdata.max())
            self._parameters_input._y_offset.set_value(x0=xdata.mean(),
                                                       k0=0,
                                                       c0=ydata.min(),
                                                       I0=ydata.max())

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

        def _band_tree_callback():
            self._spectra_plot.cls_line_intensity()
            self._spectra_plot.cls_texts()
            for _band, _v_upper, _v_lower, _branch in self._branch_tree.get_select_state():
                self.plot_line_intensity(band=_band, v_upper=_v_upper, v_lower=_v_lower,
                                         branch=_branch)

        self._spectra_plot.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self._file_read.TextChangedSignal.connect(_file_read_callback)
        self._parameters_input.valueChanged.connect(_parameters_input_callback)
        self._wavelength_range.valueChanged.connect(_parameters_input_callback)
        self._resize_input.valueChanged.connect(lambda: _resize_plot(self._resize_input.value()))
        self._branch_tree.stateChanged.connect(_band_tree_callback)
        self._plot_buttons['add_sim'].clicked.connect(self.add_plot_sim)
        self._plot_buttons['fit'].clicked.connect(self.sim_exp)
        self._plot_buttons['clear_exp'].clicked.connect(self._spectra_plot.cls_exp_lines)
        self._plot_buttons['clear_sim'].clicked.connect(self._spectra_plot.cls_sim_line)
        self._plot_buttons['auto_scale'].clicked.connect(self._spectra_plot.auto_scale)

    def get_sim_data(self):
        spc_func = self._spectra_tree.spectra_func
        paras_dict = self._parameters_input.value()
        Tvib = paras_dict['temperature']['Tvib']
        Trot_cold = paras_dict['temperature']['Trot_cold']
        Trot_hot = paras_dict['temperature']['Trot_hot']
        hot_ratio = paras_dict['temperature']['hot_ratio']
        wv_range = self._wavelength_range.value()
        fwhm_g = paras_dict['fwhm']['fwhm_g']
        fwhm_l = paras_dict['fwhm']['fwhm_l']

        if paras_dict['temperature']['para_form'] == 'one_Trot':
            spc_func.set_maxwell_distribution(Tvib=Tvib, Trot=Trot_cold)
        else:
            spc_func.set_double_temperature_distribution(Tvib=Tvib, Trot_hot=Trot_hot,
                                                         Trot_cold=Trot_cold, hot_ratio=hot_ratio)
        spc_func.set_intensity()
        wave_exp = self._exp_data['wavelength']
        wave_in_range = wave_exp[np.logical_and(wave_exp < wv_range[1], wave_exp > wv_range[0])]
        self._x_correct_func = self._parameters_input._x_offset.correct_func(**paras_dict[
            'x_offset'])
        self._x_correct_reversed_func = self._parameters_input._x_offset.correct_func_reversed(
                **paras_dict['x_offset'])
        wave_range_corrected = self._x_correct_func(wv_range)
        wavelength_corrected = self._x_correct_func(wave_exp)

        kwargs = dict(wavelength_range=wave_range_corrected,
                      waveLength_exp=wavelength_corrected,
                      fwhm=dict(Gaussian=fwhm_g, Lorentzian=fwhm_l),
                      slit_func=paras_dict['fwhm']['para_form'],
                      normalized=True)

        wv_in_range_corrected, intensity_normalized = spc_func.get_extended_wavelength(**kwargs)
        self._normalized_factor = spc_func.normalized_factor
        self._y_correct_func = self._parameters_input._y_offset.correct_func(
                **paras_dict['y_offset'])
        intensity_correct = self._y_correct_func(wv_in_range_corrected, intensity_normalized)
        ##
        # if not self._normalized_groupbox.is_nomalized():
        #     intensity_correct = intensity_correct * self._normalized_groupbox.value()
        # print(self._normalized_factor)
        ##
        return wave_in_range, intensity_correct

    def add_plot_sim(self):
        xdata, ydata = self.get_sim_data()
        self._spectra_plot.set_sim_line(xdata=xdata,
                                        ydata=ydata)

    def plot_sim(self):
        self._spectra_plot.cls_sim_line()
        self.add_plot_sim()

    def _get_line_intensity(self, band, v_upper, v_lower, branch):
        paras_dict = self._parameters_input.value()
        Tvib = paras_dict['temperature']['Tvib']
        Trot_cold = paras_dict['temperature']['Trot_cold']
        Trot_hot = paras_dict['temperature']['Trot_hot']
        hot_ratio = paras_dict['temperature']['hot_ratio']
        Trot_para_form = paras_dict['temperature']['para_form']

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
        return self._x_correct_reversed_func(wv), self._y_correct_func(self._exp_data[
                                                                           'wavelength'],
                                                                       intens /
                                                                       self._normalized_factor)

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

    def sim_exp(self):
        _spc_func = self._spectra_tree.spectra_func

        paras_dict = self._parameters_input.value()
        Tvib_init = paras_dict['temperature']['Tvib']
        Trot_cold_init = paras_dict['temperature']['Trot_cold']
        Trot_hot_init = paras_dict['temperature']['Trot_hot']
        hot_ratio_init = paras_dict['temperature']['hot_ratio']
        fwhm_g_init = paras_dict['fwhm']['fwhm_g']
        fwhm_l_init = paras_dict['fwhm']['fwhm_l']

        wave_exp = self._exp_data['wavelength']
        intens_exp = self._exp_data['intensity']
        wv_range = self._wavelength_range.value()

        _bool = np.logical_and(wave_exp < wv_range[1], wave_exp > wv_range[0])
        wave_in_range = wave_exp[_bool]
        intens_in_range = intens_exp[_bool]
        #
        _state = self._parameters_input.state()
        x_offset_x0 = self._parameters_input.value()['x_offset']['x0']
        slit_func_name = self._parameters_input.value()['fwhm']['para_form']

        Trot_para_form = paras_dict['temperature']['para_form']

        def tracer(Tvib, Trot, fwhm_g, fwhm_l):
            _str_0 = 'Tvib={Tvib:4.0f} K, Trot={Trot:4.0f} K, '.format(Tvib=Tvib, Trot=Trot)
            _str_1 = 'fwhm_g={g:.3f} nm, fwhm_l={l:.3f} nm'.format(g=fwhm_g, l=fwhm_l)
            print(_str_0 + _str_1)

        def func(x, Tvib, Trot_cold, Trot_hot, hot_ratio, fwhm_g, fwhm_l,
                 x_offset_k0, x_offset_k1, x_offset_k2, x_offset_k3,
                 y_offset_x0, y_offset_k0, y_offset_c0, y_offset_I0):
            tracer(Tvib, Trot_cold, fwhm_g, fwhm_l)

            if Trot_para_form == 'one_Trot':
                _spc_func.set_maxwell_distribution(Tvib=Tvib, Trot=Trot_cold)
            else:
                _spc_func.set_double_temperature_distribution(Tvib=Tvib,
                                                              Trot_cold=Trot_cold,
                                                              Trot_hot=Trot_hot,
                                                              hot_ratio=hot_ratio)

            _spc_func.set_intensity()

            x_correct_func_kwargs = dict(x0=x_offset_x0,
                                         k0=x_offset_k0, k1=x_offset_k1,
                                         k2=x_offset_k2, k3=x_offset_k3)
            y_correct_func_kwargs = dict(x0=y_offset_x0,
                                         k0=y_offset_k0,
                                         c0=y_offset_c0,
                                         I0=y_offset_I0)
            x_correct_func = self._parameters_input._x_offset.correct_func(
                    **x_correct_func_kwargs)
            y_correct_func = self._parameters_input._y_offset.correct_func(**y_correct_func_kwargs)
            wave_range_corrected = x_correct_func(wv_range)
            wavelength_corrected = x_correct_func(x)

            _, intens = _spc_func.get_extended_wavelength(wavelength_range=wave_range_corrected,
                                                          waveLength_exp=wavelength_corrected,
                                                          slit_func=slit_func_name,
                                                          fwhm={'Gaussian': fwhm_g,
                                                                'Lorentzian': fwhm_l},
                                                          normalized=True)
            return y_correct_func(_, intens)

        # build model
        spectra_fit_model = Model(func)
        params = spectra_fit_model.make_params()
        init_value = dict(Tvib=Tvib_init,
                          Trot_hot=Trot_hot_init, Trot_cold=Trot_cold_init,
                          hot_ratio=hot_ratio_init,
                          fwhm_g=fwhm_g_init, fwhm_l=fwhm_l_init,
                          x_offset_k0=paras_dict['x_offset']['k0'],
                          x_offset_k1=paras_dict['x_offset']['k1'],
                          x_offset_k2=paras_dict['x_offset']['k2'],
                          x_offset_k3=paras_dict['x_offset']['k3'],
                          y_offset_x0=paras_dict['y_offset']['x0'],
                          y_offset_k0=paras_dict['y_offset']['k0'],
                          y_offset_c0=paras_dict['y_offset']['c0'],
                          y_offset_I0=paras_dict['y_offset']['I0'])
        temp_state = self._parameters_input._temperature.state()
        varied_variable = dict(Tvib=_state['temperature'] and temp_state[0],
                               Trot_cold=_state['temperature'] and temp_state[1],
                               Trot_hot=_state['temperature'] and temp_state[2],
                               hot_ratio=_state['temperature'] and temp_state[3],
                               fwhm_g=_state['fwhm'],
                               fwhm_l=_state['fwhm'],
                               x_offset_k0=self._parameters_input._x_offset.state()[1],
                               x_offset_k1=self._parameters_input._x_offset.state()[2],
                               x_offset_k2=self._parameters_input._x_offset.state()[3],
                               x_offset_k3=self._parameters_input._x_offset.state()[4],
                               y_offset_x0=False,
                               y_offset_k0=self._parameters_input._y_offset.state()[1],
                               y_offset_c0=self._parameters_input._y_offset.state()[2],
                               y_offset_I0=self._parameters_input._y_offset.state()[3])

        range_dict = dict(Tvib=(300, 10000),
                          Trot_cold=(300, 10000),
                          Trot_hot=(300, 20000),
                          hot_ratio=(0, 1),
                          fwhm_g=(0, 1),
                          fwhm_l=(0, 1),
                          x_offset_k0=(-np.inf, np.inf),
                          x_offset_k1=(-np.inf, np.inf),
                          x_offset_k2=(-np.inf, np.inf),
                          x_offset_k3=(-np.inf, np.inf),
                          y_offset_x0=(-np.inf, np.inf),
                          y_offset_k0=(-np.inf, np.inf),
                          y_offset_c0=(-np.inf, np.inf),
                          y_offset_I0=(0, np.inf))

        for key in init_value:
            params[key].set(value=init_value[key])
            params[key].set(vary=varied_variable[key])
            params[key].set(min=range_dict[key][0], max=range_dict[key][1])

        self._sim_result = spectra_fit_model.fit(intens_in_range, params=params,
                                                 # method='least_squares',
                                                 fit_kws=dict(ftol=1e-12,
                                                              xtol=1e-12),
                                                 x=wave_in_range)
        self._spectra_plot.cls_sim_line()
        self._spectra_plot.set_sim_line(xdata=wave_in_range,
                                        ydata=self._sim_result.best_fit)
        self._parameters_input.set_value(**self._sim_result.values)
        # QClipboard().setText('copy_trext')
        cb = QW.QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText("Clipboard Text", mode=cb.Clipboard)
        self.show_report()

    def mouse_move(self, event):
        if event.inaxes:
            self.statusBar().showMessage('x={x:.2f}, y={y:.2f}'.format(x=event.xdata,
                                                                       y=event.ydata))
        else:
            self.statusBar().showMessage('Ready')

    def set_button_layout(self):
        self.button_layout = QW.QVBoxLayout()
        sub_layout = QW.QGridLayout()
        self._plot_buttons = dict()
        self._plot_buttons['clear_sim'] = BetterButton('ClearSim')
        self._plot_buttons['clear_exp'] = BetterButton('ClearExp')
        self._plot_buttons['add_sim'] = BetterButton('AddSim')
        # self._plot_buttons['add_exp'] = BetterButton('Add&Exp')
        self._plot_buttons['auto_scale'] = BetterButton('&AutoScale')
        self._plot_buttons['fit'] = BetterButton('&Fit')
        sub_layout.addWidget(self._plot_buttons['auto_scale'], 0, 0)
        sub_layout.addWidget(self._plot_buttons['add_sim'], 1, 0)
        # sub_layout.addWidget(self._plot_buttons['add_exp'], 2, 0)
        sub_layout.addWidget(self._plot_buttons['clear_sim'], 2, 0)
        sub_layout.addWidget(self._plot_buttons['clear_exp'], 3, 0)
        sub_layout.addWidget(self._plot_buttons['fit'], 4, 0, 2, 1)
        # sub_layout.addWidget(self._no)
        self.button_layout.addLayout(sub_layout)
        self.button_layout.addStretch(1)

    def show_report(self):
        _str = '' if self._sim_result is None else self._sim_result.fit_report()
        msg = QW.QMessageBox()
        msg.setIcon(QW.QMessageBox.Information)
        current_time = time.strftime('%Y/%m/%d %H:%M:%S')
        msg.setText(current_time)
        msg.setInformativeText(_str)
        msg.setFont(QFont('Consolas', 12))
        msg.setWindowTitle('Simulation Report')
        msg.setStandardButtons(QW.QMessageBox.Save | QW.QMessageBox.Close)
        msg.exec_()
        # msg.buttonClicked.connect(btn_callback)


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
    app.setStyle(QW.QStyleFactory.create('Fusion'))
    window = GUISpectra()
    window.show()
    app.exec_()
    # run_app()
