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
import time
from spectra import OHSpectra
from lmfit import Model
from PyQt5 import QtWidgets as QW
from PyQt5.QtGui import QIcon, QFont
from BetterQWidgets import BetterQPushButton
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

        init_width, init_height = 15, 5
        self._spectra_plot = SpectraPlot(self.cenWidget, width=init_width, height=init_height)
        self._file_read = ReadFileQWidget()
        self._spectra_tree = SpectraFunc()
        self._wavelength_range = RangeQWidget()
        self._parameters_input = ParaQWidget()
        self._report = ReportQWidget()
        self._branch_tree = BandBranchesQTreeWidget()
        self._resize_input = SizeInput(init_width=init_width, init_height=init_height)
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
        report_action = QW.QAction('Report', self)
        save_action.setFont(QFont('Ubuntu', 15))
        report_action.setFont(QFont('Ubuntu', 15))
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
            # self._spectra_plot.auto_scale()
            # self._parameters_input._y_offset.set_value(x0=xdata.mean(),
            #                                            k0=0,
            #                                            c0=ydata.min(),
            #                                            I0=ydata.max())

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
        # self._wavelength_range.valueChanged.connect(_parameters_input_callback)
        self._resize_input.valueChanged.connect(lambda: _resize_plot(self._resize_input.value()))
        self._branch_tree.stateChanged.connect(_band_tree_callback)
        self._plot_buttons['add_sim'].clicked.connect(self.add_plot_sim)
        self._plot_buttons['fit'].clicked.connect(self.sim_exp)
        self._plot_buttons['clear_exp'].clicked.connect(self._spectra_plot.cls_exp_lines)
        self._plot_buttons['clear_sim'].clicked.connect(self._spectra_plot.cls_sim_line)
        self._plot_buttons['auto_scale'].clicked.connect(self._spectra_plot.auto_scale)

    def get_sim_data(self):
        spc_func = self._spectra_tree.spectra_func
        wv_range = self._wavelength_range.value()
        slit_func = self._parameters_input._fwhm.para_form()
        intensity_correct = self._evolve_spectra(spc_func, wv_range, slit_func,
                                                 self._exp_data['wavelength'],
                                                 *self._parameters_input.value())

        wave_exp = self._exp_data['wavelength']
        wave_in_range = wave_exp[np.logical_and(wave_exp < wv_range[1], wave_exp > wv_range[0])]
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

    def _evolve_spectra(self, _spc_func, wv_range, slit_func_name,
                        x, Tvib, Trot_cold, Trot_hot,
                        hot_ratio, fwhm_g, fwhm_l,
                        x_offset_x0, x_offset_k0, x_offset_k1, x_offset_k2, x_offset_k3,
                        y_offset_x0, y_offset_k0, y_offset_c0, y_offset_I0):
        # --------------------------------------------------------------------------------------- #
        # tracer
        _str_0 = 'Tvib={Tvib:4.0f} K, Trot={Trot:4.0f} K, '.format(Tvib=Tvib, Trot=Trot_cold)
        _str_1 = 'fwhm_g={g:.3f} nm, fwhm_l={l:.3f} nm'.format(g=fwhm_g, l=fwhm_l)
        print(_str_0 + _str_1)
        # --------------------------------------------------------------------------------------- #
        if self._parameters_input._temperature.para_form() == 'one_Trot':
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

    def sim_exp(self):
        _spc_func = self._spectra_tree.spectra_func
        wave_exp = self._exp_data['wavelength']
        intens_exp = self._exp_data['intensity']
        wv_range = self._wavelength_range.value()
        _bool = np.logical_and(wave_exp < wv_range[1], wave_exp > wv_range[0])
        wave_in_range = wave_exp[_bool]
        intens_in_range = intens_exp[_bool]
        #
        slit_func_name = self._parameters_input._fwhm.para_form()

        def func(x, Tvib, Trot_cold, Trot_hot, hot_ratio, fwhm_g, fwhm_l,
                 x_offset_x0, x_offset_k0, x_offset_k1, x_offset_k2, x_offset_k3,
                 y_offset_x0, y_offset_k0, y_offset_c0, y_offset_I0):
            return self._evolve_spectra(_spc_func, wv_range, slit_func_name,
                                        x, Tvib, Trot_cold, Trot_hot, hot_ratio,
                                        fwhm_g, fwhm_l,
                                        x_offset_x0, x_offset_k0, x_offset_k1, x_offset_k2,
                                        x_offset_k3,
                                        y_offset_x0, y_offset_k0, y_offset_c0, y_offset_I0)

        # build model
        spectra_fit_model = Model(func)
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
        # cb = QW.QApplication.clipboard()
        # cb.clear(mode=cb.Clipboard)
        # cb.setText("Clipboard Text", mode=cb.Clipboard)
        self._goodness_of_fit.set_value(p_data=self._sim_result.best_fit,
                                        o_data=self._sim_result.data)
        self.print_sim_result()
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
        self._plot_buttons['clear_sim'] = BetterQPushButton('ClearSim')
        self._plot_buttons['clear_exp'] = BetterQPushButton('ClearExp')
        self._plot_buttons['add_sim'] = BetterQPushButton('AddSim')
        # self._plot_buttons['add_exp'] = BetterButton('Add&Exp')
        self._plot_buttons['auto_scale'] = BetterQPushButton('&AutoScale')
        self._plot_buttons['fit'] = BetterQPushButton('&Fit')
        sub_layout.addWidget(self._plot_buttons['auto_scale'], 0, 0)
        sub_layout.addWidget(self._plot_buttons['add_sim'], 1, 0)
        # sub_layout.addWidget(self._plot_buttons['add_exp'], 2, 0)
        sub_layout.addWidget(self._plot_buttons['clear_sim'], 2, 0)
        sub_layout.addWidget(self._plot_buttons['clear_exp'], 3, 0)
        sub_layout.addWidget(self._plot_buttons['fit'], 4, 0, 2, 1)
        # sub_layout.addWidget(self._no)
        self.button_layout.addLayout(sub_layout)
        self.button_layout.addStretch(1)

    def simulated_result_to_copy(self):
        def get_print_str(param, _format):
            value = param.value
            if param.vary:
                return '\n{v:{frmt}} {err:{frmt}}'.format(v=value, err=param.stderr,
                                                          frmt=_format)
            else:
                return '\n{v:{frmt}} [fixed]'.format(value=value, frmt=_format)

        _str = ''
        _str += get_print_str(self._sim_result.params['Tvib'], '.0f')
        _str += get_print_str(self._sim_result.params['Trot_cold'], '.0f')
        _str += get_print_str(self._sim_result.params['fwhm_g'], '.3f')
        _str += get_print_str(self._sim_result.params['fwhm_l'], '.3f')
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
        msg.setStandardButtons(QW.QMessageBox.Save | QW.QMessageBox.Close)
        msg.exec_()
        # msg.buttonClicked.connect(btn_callback)

    def print_sim_result(self):
        output = []
        _str = '{value:.3e} {stderr:.3e} '
        for _ in ('Tvib', 'Trot_cold', 'fwhm_g', 'fwhm_l'):
            print(_)
            output.append(_str.format(value=self._sim_result.params[_].value,
                                      stderr=self._sim_result.params[_].stderr))
        output.append('{r2:.4f}'.format(r2=self._goodness_of_fit._r2))
        print(''.join(output))


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
