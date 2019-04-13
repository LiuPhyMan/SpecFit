#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 12:31 2018/4/7

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   test
@IDE:       PyCharm
"""
import sys
import re
import numpy as np
from numpy.polynomial import Polynomial as P
from matplotlib import ticker
from PyQt5 import QtWidgets as QW
from PyQt5.QtGui import QCursor, QFont
from PyQt5.QtCore import Qt, QSize, pyqtSignal
from spectra import OHSpectra, COSpectra, N2Spectra, AddSpectra, convolute_to_voigt
from .widgets import QPlot
from BetterQWidgets import BetterQLabel, BetterQDoubleSpinBox, SciQDoubleSpinBox
from BetterQWidgets import BetterQPushButton as BetterButton

_GROUPBOX_TITLE_STYLESHEET = "QGroupBox {font-weight: bold; font-family: Helvetica; font-size: " \
                             "10pt}"
_DOUBLESPINBOX_STYLESHEET = 'QDoubleSpinBox:disabled {color : rgb(210, 210, 210)}'
_LABEL_STYLESHEET = 'QLabel:disabled {color : rgb(210, 210, 210)}'

_DEFAULT_FONT = QFont("Helvetica", 10)
_DOUBLESPINBOX_FONT = QFont('Helvetica', 11)
_LABEL_FONT = QFont("Helvetica", 12)


class _DefaultQGroupBox(QW.QGroupBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(_GROUPBOX_TITLE_STYLESHEET)
        self.setCheckable(True)
        self.setChecked(True)


class _DefaultQDoubleSpinBox(QW.QDoubleSpinBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(_DOUBLESPINBOX_STYLESHEET)
        self.setFont(_DOUBLESPINBOX_FONT)
        self.setFixedWidth(85)
        self.setAlignment(Qt.AlignRight)
        self.setButtonSymbols(self.NoButtons)


class _DefaultQLabel(QW.QLabel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet(_LABEL_STYLESHEET)
        self.setFont(_LABEL_FONT)
        self.setAlignment(Qt.AlignRight)


class _DefaultQComboBox(QW.QComboBox):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFont(_DEFAULT_FONT)
        self.setCursor(Qt.PointingHandCursor)


class TemperatureQGroupBox(_DefaultQGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("TEMPERATURE")
        self._vib_input = _DefaultQDoubleSpinBox()
        self._rot_cold_input = _DefaultQDoubleSpinBox()
        self._rot_hot_input = _DefaultQDoubleSpinBox()
        self._hot_ratio = _DefaultQDoubleSpinBox()
        self._distribution_combobox = _DefaultQComboBox()
        self._labels = dict()
        self._set_labels()
        self._set_spinbox()
        self._set_combobox()
        self._set_layout()
        self._set_slot()

    def para_form(self):
        if self._distribution_combobox.currentText() == 'one_Trot':
            return 'one_Trot'
        else:
            return 'two_Trot'

    def value(self):
        return dict(para_form=self._distribution_combobox.currentText(),
                    Tvib=self._vib_input.value(),
                    Trot_cold=self._rot_cold_input.value(),
                    Trot_hot=self._rot_hot_input.value(),
                    hot_ratio=self._hot_ratio.value())

    def set_value(self, *, Tvib, Trot_cold, Trot_hot, hot_ratio):
        self._vib_input.setValue(Tvib)
        self._rot_cold_input.setValue(Trot_cold)
        self._rot_hot_input.setValue(Trot_hot)
        self._hot_ratio.setValue(hot_ratio)

    def state(self):
        if self.para_form() == 'one_Trot':
            return [True, True, False, False]
        else:
            return [True, True, True, True]

    def is_variable_state(self):
        if self.isChecked():
            return self.state()
        else:
            return [False for _ in self.state()]

    def _set_labels(self):
        for _ in ("Tvib", "Trot_cold", "Trot_hot", "hot_ratio"):
            self._labels[_] = _DefaultQLabel(_)
        self._labels["Tvib"] = _DefaultQLabel("<b><i>T</i></b><sub>vib</sub>")
        self._labels["Trot_cold"] = _DefaultQLabel("<b><i>T</i></b><sub>rot</sub>")
        self._labels["Trot_hot"] = _DefaultQLabel("<b><i>T</b></i><sub>rot, hot</sub>")
        self._labels["hot_ratio"] = _DefaultQLabel("<b><i>r</b></i><sub>hot</sub>")

    def _set_layout(self):
        _layout = QW.QFormLayout()
        _layout.addRow(self._labels["Tvib"], self._vib_input)
        _layout.addRow(self._labels["Trot_cold"], self._rot_cold_input)
        _layout.addRow(self._labels["Trot_hot"], self._rot_hot_input)
        _layout.addRow(self._labels["hot_ratio"], self._hot_ratio)
        _layout.addRow("", self._distribution_combobox)
        _layout.setLabelAlignment(Qt.AlignRight)
        self.setLayout(_layout)

    def _set_spinbox(self):
        #               range[0] range[1] step value suffix decimals
        setting_dict = [(300, 1e4, 10, 3000, ' K', 0),  # Tvib
                        (300, 2e4, 10, 3000, ' K', 0),  # Trot_cold
                        (300, 2e4, 50, 3000, ' K', 0),  # Trot_hot
                        (0, 1, 0.05, 0.5, '', 2)]  # hot_ratio
        for _, setting in zip((self._vib_input,
                               self._rot_cold_input,
                               self._rot_hot_input,
                               self._hot_ratio), setting_dict):
            _.setRange(setting[0], setting[1])
            _.setSingleStep(setting[2])
            _.setValue(setting[3])
            _.setSuffix(setting[4])
            _.setDecimals(setting[5])

    def _set_combobox(self):
        self._distribution_combobox.addItem('one_Trot')
        self._distribution_combobox.addItem('two_Trot')

    def _set_slot(self):
        def slot_emit():
            self.valueChanged.emit()

        def _combobox_callback():
            self._labels['Tvib'].setEnabled(self.state()[0])
            self._labels['Trot_cold'].setEnabled(self.state()[1])
            self._labels['Trot_hot'].setEnabled(self.state()[2])
            self._labels['hot_ratio'].setEnabled(self.state()[3])
            self._vib_input.setEnabled(self.state()[0])
            self._rot_cold_input.setEnabled(self.state()[1])
            self._rot_hot_input.setEnabled(self.state()[2])
            self._hot_ratio.setEnabled(self.state()[3])
            slot_emit()

        self._vib_input.valueChanged.connect(slot_emit)
        self._rot_cold_input.valueChanged.connect(slot_emit)
        self._rot_hot_input.valueChanged.connect(slot_emit)
        self._hot_ratio.valueChanged.connect(slot_emit)
        self._distribution_combobox.currentTextChanged.connect(_combobox_callback)
        _combobox_callback()


class FWHMQGroupBox(_DefaultQGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('FHWM')
        self.line_shape_combobox = _DefaultQComboBox()
        self.fwhm_total = _DefaultQLabel()
        self.fwhm_g_part = _DefaultQDoubleSpinBox()
        self.fwhm_l_part = _DefaultQDoubleSpinBox()
        self._label = dict(Gauss=_DefaultQLabel("<b><i>w</b></i><sub>G</sub>"),
                           Loren=_DefaultQLabel("<b><i>w</b></i><sub>L</sub>"))
        self.line_shape_combobox.addItem('Gaussian')
        self.line_shape_combobox.addItem('Lorentzian')
        self.line_shape_combobox.addItem('Voigt')
        self.line_shape_combobox.setCurrentText('Voigt')

        self._set_format()
        self._set_layout()
        self._set_slot()

    def para_form(self):
        return self.line_shape_combobox.currentText()

    def value(self):
        return dict(para_form=self.para_form(),
                    fwhm_g=self.fwhm_g_part.value() * 1e-3,
                    fwhm_l=self.fwhm_l_part.value() * 1e-3)

    def state(self):
        if self.para_form() == 'Gaussian':
            return [True, False]
        elif self.para_form() == 'Lorentzian':
            return [False, True]
        elif self.para_form() == 'Voigt':
            return [True, True]

    def is_variable_state(self):
        if self.isChecked():
            return self.state()
        else:
            return [False for _ in self.state()]

    def set_value(self, *, fwhm_g, fwhm_l):
        self.fwhm_g_part.setValue(fwhm_g * 1e3)
        self.fwhm_l_part.setValue(fwhm_l * 1e3)

    def _set_format(self):
        font = QFont("Helvetica", 13)
        font.setUnderline(True)
        self.fwhm_total.setFont(font)

        for _fwhm in (self.fwhm_g_part, self.fwhm_l_part):
            _fwhm.setSuffix(" pm")
            _fwhm.setRange(0, 1e3)
            _fwhm.setDecimals(2)
            _fwhm.setSingleStep(0.5)
            _fwhm.setAccelerated(True)
            _fwhm.setValue(20)

    def _set_layout(self):
        _layout = QW.QFormLayout()
        _layout.addRow(self.fwhm_total)
        _layout.addRow(self._label["Gauss"], self.fwhm_g_part)
        _layout.addRow(self._label["Loren"], self.fwhm_l_part)
        _layout.addRow("", self.line_shape_combobox)
        self.setLayout(_layout)

    def _set_slot(self):
        self.line_shape_combobox.currentTextChanged.connect(self.line_shape_combobox_callback)
        self.fwhm_g_part.valueChanged.connect(lambda: self.valueChanged.emit())
        self.fwhm_l_part.valueChanged.connect(lambda: self.valueChanged.emit())
        self.valueChanged.connect(self.set_fwhm_total)
        self.line_shape_combobox_callback()

    def line_shape_combobox_callback(self):
        def set_fwhm_g_state(state):
            self.fwhm_g_part.setEnabled(state)
            self._label['Gauss'].setEnabled(state)

        def set_fwhm_l_state(state):
            self.fwhm_l_part.setEnabled(state)
            self._label['Loren'].setEnabled(state)

        _state_dict = dict(Gaussian=(True, False),
                           Lorentzian=(False, True),
                           Voigt=(True, True))
        set_fwhm_g_state(_state_dict[self.para_form()][0])
        set_fwhm_l_state(_state_dict[self.para_form()][1])
        self.valueChanged.emit()

    def set_fwhm_total(self):
        if self.para_form() == 'Gaussian':
            _value = self.value()['fwhm_g']
        elif self.para_form() == 'Lorentzian':
            _value = self.value()['fwhm_l']
        elif self.para_form() == 'Voigt':
            _value = convolute_to_voigt(fwhm_G=self.value()['fwhm_g'],
                                        fwhm_L=self.value()['fwhm_l'])
        self.fwhm_total.setText('<i><b>{i:.2f}</b><i> pm'.format(i=_value * 1e3))


class XOffsetQGroupBox(_DefaultQGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('X-OFFSET')

        self._combobox = _DefaultQComboBox()
        self._spinBoxes = dict()
        self._label = dict()
        self._set_combobox()
        self._set_labels()
        self._set_spinBoxes()
        self._set_layout()
        self._set_slot()

    def para_form(self):
        return self._combobox.currentText().lower()

    def value(self):
        return dict(para_form=self.para_form(),
                    x0=self._spinBoxes['x0'].value(),
                    k0=self._spinBoxes['k0'].value(),
                    k1=self._spinBoxes['k1'].value(),
                    k2=self._spinBoxes['k2'].value(),
                    k3=self._spinBoxes['k3'].value())

    def set_value(self, *, x0, k0, k1, k2, k3):
        self._spinBoxes['x0'].setValue(x0)
        self._spinBoxes['k0'].setValue(k0)
        self._spinBoxes['k1'].setValue(k1)
        self._spinBoxes['k2'].setValue(k2)
        self._spinBoxes['k3'].setValue(k3)

    def state(self):
        if self._combobox.currentText().lower() == 'constant':
            state = [1, 1, 0, 0, 0]
        if self._combobox.currentText().lower() == 'linear':
            state = [1, 1, 1, 0, 0]
        if self._combobox.currentText().lower() == 'parabolic':
            state = [1, 1, 1, 1, 0]
        if self._combobox.currentText().lower() == 'cubic':
            state = [1, 1, 1, 1, 1]
        return [True if _ else False for _ in state]

    def is_variable_state(self):
        if self.isChecked():
            return [False] + self.state()[1:]
        else:
            return [False for _ in self.state()]

    def correct_func(self, **kwargs):
        x0 = kwargs['x0']
        p = P([kwargs[key] for key in ('k0', 'k1', 'k2', 'k3')])
        if self._combobox.currentText() == 'constant'.upper():
            return lambda x: p.cutdeg(0)(x - x0) + x
        if self._combobox.currentText() == 'linear'.upper():
            return lambda x: p.cutdeg(1)(x - x0) + x
        if self._combobox.currentText() == 'parabolic'.upper():
            return lambda x: p.cutdeg(2)(x - x0) + x
        if self._combobox.currentText() == 'cubic'.upper():
            return lambda x: p.cutdeg(3)(x - x0) + x
        raise Exception('{} is error'.format(self._combobox.currentText()))

    def correct_func_reversed(self, **kwargs):
        x0 = kwargs['x0']
        if self._combobox.currentText() == 'constant'.upper():
            return lambda x: x - kwargs['k0']
        if self._combobox.currentText() == 'linear'.upper():
            return lambda x: (x - kwargs['k0'] + kwargs['k1'] * x0) / (kwargs['k1'] + 1)

    def _set_labels(self):
        for _str in ('x0', 'k0', 'k1', 'k2', 'k3'):
            self._label[_str] = _DefaultQLabel("<b><i>" + _str[:-1] + "</i></b>" +
                                               "<sub>" + _str[-1] + "</sub>")

    def _set_layout(self):
        _layout = QW.QFormLayout()
        for _str in ('x0', 'k0', 'k1', 'k2', 'k3'):
            _layout.addRow(self._label[_str], self._spinBoxes[_str])
        _layout.addRow("", self._combobox)
        self.setLayout(_layout)

    def _set_spinBoxes(self):
        _font = QFont("Helvetica", 9, italic=True)
        for key in ('x0', 'k0', 'k1', 'k2', 'k3'):
            self._spinBoxes[key] = _DefaultQDoubleSpinBox()
            self._spinBoxes[key].setRange(-np.inf, np.inf)
            self._spinBoxes[key].setSingleStep(0.002)
            self._spinBoxes[key].setDecimals(7)
            self._spinBoxes[key].setValue(0)
            self._spinBoxes[key].setAccelerated(True)
            self._spinBoxes[key].setFont(_font)
        self._spinBoxes['x0'].setSingleStep(1)
        self._spinBoxes['x0'].setDecimals(0)
        self._spinBoxes['x0'].setValue(300)
        self._spinBoxes['k1'].setValue(-0.067)

    def _set_combobox(self):
        for key in ['constant', 'linear', 'parabolic', 'cubic']:
            self._combobox.addItem(key.upper())
        self._combobox.setCurrentIndex(1)

    def _set_slot(self):
        def slot_emit():
            self.valueChanged.emit()

        def _combobox_callback():
            for _s, _l in zip(('x0', 'k0', 'k1', 'k2', 'k3'), self.state()):
                self._spinBoxes[_s].setEnabled(_l)
                self._label[_s].setEnabled(_l)
            slot_emit()

        for _key in self._spinBoxes:
            self._spinBoxes[_key].valueChanged.connect(slot_emit)

        self._combobox.currentTextChanged.connect(_combobox_callback)
        _combobox_callback()


class YOffsetQGroupBox(_DefaultQGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('Y_OFFSET')
        self.degree_combobox = _DefaultQComboBox()
        self._spinBoxes = dict()
        self._label = dict(x0=_DefaultQLabel('x0'),
                           k0=_DefaultQLabel('k0'),
                           c0=_DefaultQLabel('c0'),
                           I0=_DefaultQLabel('I0'))
        self._set_combobox()
        self._set_labels()
        self._set_spinBoxes()
        self._set_layout()
        self._set_slot()
        self.degree_combobox.setCurrentText("Even")

    def para_form(self):
        return self.degree_combobox.currentText().lower()

    def value(self):
        return dict(para_form=self.para_form(),
                    x0=self._spinBoxes['x0'].value(),
                    k0=self._spinBoxes['k0'].value(),
                    c0=self._spinBoxes['c0'].value(),
                    I0=self._spinBoxes['I0'].value())

    def state(self):
        if self.degree_combobox.currentText().lower() == 'incline':
            return [True, True, True, True]
        if self.degree_combobox.currentText().lower() == 'even':
            return [False, False, True, True]
        if self.degree_combobox.currentText().lower() == 'baselock':
            return [False, False, False, True]

    def is_variable_state(self):
        if self.isChecked():
            return [False] + self.state()[1:]
        else:
            return [False for _ in self.state()]

    def set_value(self, *, x0, k0, c0, I0):
        self._spinBoxes['x0'].setValue(x0)
        self._spinBoxes['k0'].setValue(k0)
        self._spinBoxes['c0'].setValue(c0)
        self._spinBoxes['I0'].setValue(I0)

    def correct_func(self, **kwargs):
        x0, k0, c0, I0 = kwargs['x0'], kwargs['k0'], kwargs['c0'], kwargs['I0']
        return lambda x, y: k0 * (x - x0) + c0 + (I0 - c0) * y

    def _set_labels(self):
        for _str in ('x0', 'k0', 'c0', 'I0'):
            self._label[_str] = _DefaultQLabel("<b><i>" + _str[:-1] + "</i></b>" +
                                               '<sub>' + _str[-1] + '</sub>')

    def _set_layout(self):
        _layout = QW.QFormLayout()
        for _str in ('x0', 'k0', 'c0', 'I0'):
            _layout.addRow(self._label[_str], self._spinBoxes[_str])
        _layout.addRow("", self.degree_combobox)
        self.setLayout(_layout)

    def _set_spinBoxes(self):
        _font = QFont("Helvetica", 9, italic=True)
        for key in ('x0', 'k0', 'c0', 'I0'):
            self._spinBoxes[key] = _DefaultQDoubleSpinBox()
            self._spinBoxes[key].setRange(-np.inf, np.inf)
            self._spinBoxes[key].setSingleStep(.005)
            self._spinBoxes[key].setDecimals(4)
            self._spinBoxes[key].setValue(0)
            self._spinBoxes[key].setFont(_font)
        self._spinBoxes['x0'].setDecimals(0)
        self._spinBoxes['k0'].setSingleStep(.002)
        self._spinBoxes['x0'].setValue(400)
        self._spinBoxes['x0'].setSingleStep(1)
        self._spinBoxes['I0'].setValue(1)

    def _set_combobox(self):
        self.degree_combobox.addItems(("Incline", "Even", "BaseLock"))

    def _set_slot(self):
        def slot_emit():
            self.valueChanged.emit()

        def degree_combobox_callback():
            def _set_state(term, state):
                self._spinBoxes[term].setEnabled(state)
                self._label[term].setEnabled(state)

            for _s, _l in zip(('x0', 'k0', 'c0', 'I0'), self.state()):
                _set_state(_s, _l)
            if self.degree_combobox.currentText().lower() == 'even':
                self._spinBoxes['k0'].setValue(0)
            slot_emit()

        self.degree_combobox.currentTextChanged.connect(degree_combobox_callback)
        for _key in self._spinBoxes:
            self._spinBoxes[_key].valueChanged.connect(slot_emit)

        degree_combobox_callback()


class GoodnessOfFit(_DefaultQGroupBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(False)
        self._r2 = 0
        self.setTitle("Goodness of fit")
        self._set_labels()
        self._set_layout()

    def _set_layout(self):
        layout = QW.QFormLayout()
        layout.addRow(_DefaultQLabel("<b>R</b><sup>2</sup>"), self._label_r2),
        layout.addRow(_DefaultQLabel("Chi-squared"), self._label_chisq)
        layout.setLabelAlignment(Qt.AlignRight)
        self.setLayout(layout)

    def r2(self, *, p_data, o_data):
        y_mean = np.mean(o_data)
        ss_tot = np.sum((o_data - y_mean) ** 2)
        ss_res = np.sum((o_data - p_data) ** 2)
        return 1 - ss_res / ss_tot

    def _set_labels(self):
        self._label_r2 = BetterQLabel()
        font = QFont('Ubuntu', 13)
        font.setUnderline(True)
        self._label_r2.setFont(font)
        self._label_r2.setText('0.0000')
        self._label_chisq = BetterQLabel()
        self._label_chisq.setFont(font)
        self._label_chisq.setText('0.0000')

    def set_value(self, *, p_data, o_data):
        _r2_new = self.r2(p_data=p_data, o_data=o_data)
        if self._r2 < _r2_new:
            self._label_r2.setStyleSheet("QLabel {background-color : green}")
        else:
            self._label_r2.setStyleSheet("QLabel {background-color : red}")
        self._r2 = _r2_new
        self._label_r2.setText("{:.4f}".format(self._r2))


class NormalizedQGroupBox(QW.QGroupBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(_GROUPBOX_TITLE_STYLESHEET)
        self.setCheckable(True)
        self.setTitle('Normalized')
        self._normalized_factor = 1
        self._normalized_factor_label = BetterQLabel()
        self._set_layout()
        self.set_value(self._normalized_factor)

    def is_nomalized(self):
        return self.isChecked()

    def set_value(self, value):
        self._normalized_factor = value
        self._normalized_factor_label.setText('{v:8.1f}'.format(v=value))

    def value(self):
        return self._normalized_factor

    def _set_layout(self):
        layout = QW.QVBoxLayout()
        layout.addWidget(self._normalized_factor_label)
        layout.addStretch(1)
        self.setLayout(layout)


class ParaQWidget(QW.QFrame):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._temperature = TemperatureQGroupBox()
        self._fwhm = FWHMQGroupBox()
        self._x_offset = XOffsetQGroupBox()
        self._y_offset = YOffsetQGroupBox()
        self.setFrameStyle(self.Box | self.Plain)
        self.setLineWidth(1)
        self._set_layout()
        self._set_slot()

    def state(self):
        return dict(temperature=self._temperature.isChecked(),
                    fwhm=self._fwhm.isChecked(),
                    x_offset=self._x_offset.isChecked(),
                    y_offset=self._y_offset.isChecked())

    def is_variable_state(self):
        return self._temperature.is_variable_state() + self._fwhm.is_variable_state() + \
               self._x_offset.is_variable_state() + self._y_offset.is_variable_state()

    def value(self):
        r"""
        temperature :
            Tvib, Trot
        fwhm :
            para_form, fwhm_g, fwhm_l
        x_offset :
            para_form, x0, k0, k1, k2, k3
        y_offset :
            para_form, x0, k0, c0, I0
        """
        return [self._temperature.value()['Tvib'],
                self._temperature.value()['Trot_cold'],
                self._temperature.value()['Trot_hot'],
                self._temperature.value()['hot_ratio'],
                self._fwhm.value()['fwhm_g'],
                self._fwhm.value()['fwhm_l'],
                self._x_offset.value()['x0'],
                self._x_offset.value()['k0'],
                self._x_offset.value()['k1'],
                self._x_offset.value()['k2'],
                self._x_offset.value()['k3'],
                self._y_offset.value()['x0'],
                self._y_offset.value()['k0'],
                self._y_offset.value()['c0'],
                self._y_offset.value()['I0']]

    def set_value(self, **kwargs):
        self._temperature.set_value(Tvib=kwargs['Tvib'],
                                    Trot_cold=kwargs['Trot_cold'],
                                    Trot_hot=kwargs['Trot_hot'],
                                    hot_ratio=kwargs['hot_ratio'])
        self._fwhm.set_value(fwhm_g=kwargs['fwhm_g'],
                             fwhm_l=kwargs['fwhm_l'])
        self._x_offset.set_value(x0=kwargs['x_offset_x0'],
                                 k0=kwargs['x_offset_k0'],
                                 k1=kwargs['x_offset_k1'],
                                 k2=kwargs['x_offset_k2'],
                                 k3=kwargs['x_offset_k3'])
        self._y_offset.set_value(x0=kwargs['y_offset_x0'],
                                 k0=kwargs['y_offset_k0'],
                                 c0=kwargs['y_offset_c0'],
                                 I0=kwargs['y_offset_I0'])

    def _set_layout(self):
        self.layout = QW.QGridLayout()
        self.layout.addWidget(self._temperature, 0, 0)
        self.layout.addWidget(self._fwhm, 0, 1)
        self.layout.addWidget(self._x_offset, 0, 2)
        self.layout.addWidget(self._y_offset, 0, 3)
        self.setLayout(self.layout)

    def _set_slot(self):
        def slot_emit():
            self.valueChanged.emit()

        self._temperature.valueChanged.connect(slot_emit)
        self._fwhm.valueChanged.connect(slot_emit)
        self._x_offset.valueChanged.connect(slot_emit)
        self._y_offset.valueChanged.connect(slot_emit)


class SpectraPlot(QPlot):
    _LABEL_FONTDICT = {'family': 'Helvetica', 'size': 11}
    _TEXT_FONTDICT = {'family': 'Consolas', 'size': 11}

    def __init__(self, parent=None, width=12, height=5):
        super().__init__(parent, figsize=(width, height), dpi=100,
                         toolbar_position='left')
        self.toolbar.setIconSize(QSize(16, 16))
        self.axes = self.figure.add_subplot(111)
        # self.axes.xaxis.set_major_locator(ticker.MultipleLocator(5.00))
        # self.axes.xaxis.set_minor_locator(ticker.MultipleLocator(1.00))
        # set ticker
        locator = ticker.MaxNLocator(nbins=20)
        self.axes.xaxis.set_major_locator(locator)
        self.axes.set_xlabel('Wavelength [nm]', fontdict=self._LABEL_FONTDICT)
        self.axes.set_ylabel('Intensity [a.u.]', fontdict=self._LABEL_FONTDICT)
        self.axes.set_xlim(250, 850)
        self.axes.set_ylim(-0.2, 1)

        self.axes.grid(linestyle=':')
        self.sim_line, = self.axes.plot([], [], 'r-', linewidth=.8)
        # self.exp_line, = self.axes.plot([], [], 'bo-', linewidth=.5, markersize=.5)
        self.exp_lines = []
        self._branch_lines = []
        self._texts = []
        self.figure.tight_layout()
        self.figure.canvas.mpl_connect('button_press_event', self.click_callback)
        self.temp_data = 0

    def click_callback(self, event):
        # print('{x:.2f} {y:.2f}'.format(x=event.xdata, y=event.ydata))
        print('{y0:.2f} {y1:.2f}'.format(y0=self.temp_data, y1=event.ydata))
        self.temp_data = event.ydata

    def set_sim_line(self, *, xdata, ydata):
        self.sim_line.set_data(xdata, ydata)
        self.canvas_draw()

    def cls_sim_line(self):
        self.sim_line.set_data([], [])
        self.canvas_draw()

    # def set_exp_line(self, *, xdata, ydata):
    #     self.exp_line.set_data(xdata, ydata)
    #     self.canvas_draw()
    #     self.auto_scale()
    def add_exp_line(self, *, xdata, ydata):
        self.exp_lines.append(self.axes.plot(xdata, ydata, 'bo-', linewidth=.5, markersize=.5)[0])
        self.canvas_draw()

    def hide_exp_line(self, *, index):
        self.exp_lines[index].set_visible(False)
        self.canvas_draw()

    def cls_exp_lines(self):
        for _ln in reversed(self.exp_lines):
            self.axes.lines.remove(_ln)
            del _ln
        self.exp_lines = []
        self.canvas_draw()

    def plot_line_intensity(self, *, xdata, ydata, branch='', shown_index=(), shown_J=()):
        ln, = self.axes.plot(xdata, ydata, color='magenta', linestyle='-', linewidth=1,
                             marker='o', markersize=6,
                             markerfacecolor='green', markeredgecolor='green')
        self._branch_lines.append(ln)
        for _index in shown_index:
            _text = r'{br}({J})'.format(br=branch, J=shown_J[_index])
            _ = self.axes.text(xdata[_index], ydata[_index], _text, fontsize=11, color='green',
                               style='italic', ha='center')
            self._texts.append(_)
        self.canvas_draw()

    def cls_line_intensity(self):
        for _ln in reversed(self._branch_lines):
            self.axes.lines.remove(_ln)
            del _ln
        self._branch_lines = []
        self.canvas_draw()

    def cls_texts(self):
        for _text in self._texts:
            _text.remove()
        self._texts = []
        self.canvas_draw()

    def canvas_draw(self):
        self.canvas.draw()

    def auto_scale(self):
        self.axes.autoscale(True)
        self.axes.relim()
        self.axes.autoscale_view()
        self.canvas_draw()

    def set_size(self, *, width, height):
        self.setFixedWidth(width * 100)
        self.setFixedHeight(height * 100)
        self.canvas.setFixedWidth(width * 100 * 0.98)
        self.canvas.setFixedHeight(height * 100 * 0.98)


class ReadFileQWidget(QW.QWidget):
    TextChangedSignal = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._entry = QW.QLineEdit()
        self._entry.setFont(_DEFAULT_FONT)
        self._entry.setFixedWidth(600)
        self._browse_button = BetterButton('Browse')
        self._browse_button.clicked.connect(self._get_open_file_name)
        self._set_layout()

    def _set_layout(self):
        _layout = QW.QHBoxLayout()
        _layout.addWidget(BetterQLabel('FileDir'))
        _layout.addWidget(self._entry)
        _layout.addWidget(self._browse_button)
        _layout.addStretch(1)
        self.setLayout(_layout)

    def _get_open_file_name(self):
        file_name = QW.QFileDialog.getOpenFileName(caption='Open File',
                                                   filter="data (*)")[0]
        if file_name is not None:
            self._entry.setText(file_name)
        self.TextChangedSignal.emit()


class CheckableQTreeWidget(QW.QTreeWidget):
    _DICT = dict()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.node_dict = dict()
        self.item_dict = dict()
        self.setFixedWidth(170)
        self.setCursor(QCursor(Qt.PointingHandCursor))
        self._add_dict_to_tree()
        self._set_connect()

    def _add_dict_to_tree(self):
        for _node_str in self._DICT:
            node = QW.QTreeWidgetItem(self)
            node.setText(0, _node_str)
            node.setCheckState(0, Qt.Unchecked)
            node.setFlags(node.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            self.node_dict[_node_str] = node
            self.item_dict[_node_str] = []
            for _child_str in self._DICT[_node_str]:
                child = QW.QTreeWidgetItem(node)
                child.setFlags(child.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
                child.setText(0, _child_str)
                child.setCheckState(0, Qt.Unchecked)
                self.item_dict[_node_str].append(child)

    def _add_item_to_tree(self, item_name):
        node = QW.QTreeWidgetItem(self)
        node.setText(0, item_name)
        node.setCheckState(0, Qt.Unchecked)
        node.setFlags(node.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
        self.node_dict[item_name] = node
        self.item_dict[item_name] = []

    def _set_node_check(self, node):
        self.node_dict[node].setCheckState(0, Qt.Checked)

    def _set_node_uncheck(self, node):
        self.node_dict[node].setCheckState(0, Qt.Unchecked)

    def _set_connect(self):
        def item_pressed_callback(x):
            if x.checkState(0) == Qt.Checked:
                x.setCheckState(0, Qt.Unchecked)
            elif x.checkState(0) == Qt.Unchecked:
                x.setCheckState(0, Qt.Checked)

        self.itemPressed.connect(item_pressed_callback)
        self.itemClicked.connect(lambda x: self.item_changed_callback())

    def item_changed_callback(self):
        pass

    def get_select_state(self):
        pass


class SpectraFunc(CheckableQTreeWidget):
    _DICT = {'OH(A-X) 309_nm dv=0': ['0-0 308.9 nm',
                                     '1-1 314.5 nm',
                                     '2-2 320.7 nm'],
             'OH(A-X) 283_nm dv=1': ['1-0 282.8 nm',
                                     '2-1 289.1 nm',
                                     '3-2 296.1 nm'],
             'CO(B-A)': ['0-0 451.1 nm',
                         '0-1 483.5 nm',
                         '0-2 519.8 nm',
                         '0-3 561.0 nm',
                         '0-4 608.0 nm',
                         '0-5 662.0 nm',
                         '1-0 412.4 nm',
                         '1-1 439.3 nm',
                         '1-2 469.7 nm'],
             "N2(C-B) 316_nm dv=1 ": ['1-0 315.8 nm',
                                      '2-1 313.5 nm',
                                      '3-2 311.5 nm'],
             'N2(C-B) 337_nm dv=0': ['0-0 337.0 nm',
                                     '1-1 333.8 nm',
                                     '2-2 330.9 nm'],
             'N2(C-B) 358_nm dv=-1': ['0-1 357.6 nm',
                                      '1-2 353.6 nm',
                                      '2-3 349.9 nm',
                                      '3-4 346.8 nm'],
             "N2(C-B) 380_nm dv=-2": ['0-2 380.4 nm',
                                      '1-3 375.4 nm',
                                      '2-4 370.9 nm'],
             "N2(C-B) 406_nm dv=-3": ['0-3 405.8 nm',
                                      '1-4 399.7 nm',
                                      '2-5 394.2 nm',
                                      '3-6 389.4 nm'],
             "N2+(B-X) 358_nm dv=1": ['1-0 357.9 nm',
                                      '2-1 356.1 nm',
                                      '3-2 354.6 nm',
                                      '4-3 353.5 nm'],
             "N2+(B-X) 391_nm dv=0": ['0-0 391.1 nm',
                                      '1-1 388.1 nm',
                                      '2-2 385.5 nm',
                                      '3-3 383.2 nm'],
             "N2+(B-X) 427_nm dv=-1": ['0-1 427.5 nm',
                                       '1-2 423.3 nm',
                                       '2-3 419.6 nm',
                                       '3-4 416.4 nm']}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spectra_func = None
        self.setFixedWidth(220)
        self.setFixedHeight(500)
        self._set_node_check('OH(A-X) 309_nm dv=0')
        self.item_changed_callback()

    def item_changed_callback(self):
        spectra_list = self.get_select_state()
        print(spectra_list)

        def get_spectra(_spectra):
            temp = re.fullmatch(r"([^()]+)\(([^()]+)\)_(\d+)-(\d+)", _spectra).groups()
            molecule, band, v_upper_str, v_lower_str = temp
            band_dict = dict(OH=OHSpectra,
                             CO=COSpectra,
                             N2=N2Spectra)
            func = band_dict[molecule]
            return func(band=band, v_upper=int(v_upper_str), v_lower=int(v_lower_str))

        if len(spectra_list) == 1:
            self.spectra_func = get_spectra(spectra_list[0])
        if len(spectra_list) > 1:
            self.spectra_func = get_spectra(spectra_list[0])
            for i in range(1, len(spectra_list)):
                self.spectra_func = AddSpectra(self.spectra_func, get_spectra(spectra_list[i]))

    def get_select_state(self):
        selected_list = []
        for key in self._DICT:
            for _ in self.item_dict[key]:
                if _.checkState(0) == Qt.Checked:
                    spec = re.match(r"([^()]+)\(([^()]+)\)", key).group()
                    selected_list.append('{spec}_{band}'.format(spec=spec, band=_.text(0)[:3]))
        return selected_list


class BandBranchesQTreeWidget(CheckableQTreeWidget):
    _DICT = {'OH(A-X)_0-0': OHSpectra._BRANCH_SEQ,
             'OH(A-X)_1-1': OHSpectra._BRANCH_SEQ,
             'OH(A-X)_2-2': OHSpectra._BRANCH_SEQ,
             'OH(A-X)_1-0': OHSpectra._BRANCH_SEQ,
             'OH(A-X)_2-1': OHSpectra._BRANCH_SEQ,
             'OH(A-X)_3-2': OHSpectra._BRANCH_SEQ}
    stateChanged = pyqtSignal()

    def __int__(self, parent=None):
        super().__int__(parent)
        self.setFixedWidth(200)
        self.setFixedHeight(500)

    def item_changed_callback(self):
        self.stateChanged.emit()

    def get_select_state(self):
        selected_list = []
        for band in self._DICT:
            for branch in self.item_dict[band]:
                if branch.checkState(0) == Qt.Checked:
                    v_upper = int(band[-3])
                    v_lower = int(band[-1])
                    selected_list.append([band[:-4], v_upper, v_lower, branch.text(0)])
        return selected_list


class ExpLinesQTreeWidget(CheckableQTreeWidget):
    _DICT = {}
    stateChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setFixedHeight(500)

    def item_changed_callback(self):
        self.stateChanged.emit()

    def add_exp(self, name):
        self._add_item_to_tree(name)

    def remove_all(self):
        self.clear()


class SizeInput(QW.QDialog):
    valueChanged = pyqtSignal()

    def __init__(self, init_width, init_height, parent=None):
        super().__init__(parent)
        # self._width_spinbox = BetterQDoubleSpinBox()
        self._width_spinbox = _DefaultQDoubleSpinBox()
        self._height_spinbox = BetterQDoubleSpinBox()
        self._width_spinbox.setValue(init_width)
        self._height_spinbox.setValue(init_height)
        self._set_layout()
        self._set_slot()

    def value(self):
        return self._width_spinbox.value(), self._height_spinbox.value()

    def _set_layout(self):
        _layout = QW.QFormLayout()
        _layout.addRow(_DefaultQLabel("Width"), self._width_spinbox)
        _layout.addRow(_DefaultQLabel("Height"), self._height_spinbox)
        self.setLayout(_layout)

    def _set_slot(self):
        self._width_spinbox.valueChanged.connect(lambda: self.valueChanged.emit())
        self._height_spinbox.valueChanged.connect(lambda: self.valueChanged.emit())


class FitArgsInput(QW.QDialog):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # self._ftol_spinbox = BetterQDoubleSpinBox()
        self._ftol_spinbox = SciQDoubleSpinBox(node_range=(-15, -1))
        # self._xtol_spinbox = BetterQDoubleSpinBox()
        self._xtol_spinbox = SciQDoubleSpinBox(node_range=(-15, -1))
        self._ftol_spinbox.setValue(1e-8)
        self._xtol_spinbox.setValue(1e-8)
        self._set_layout()
        self._set_slot()

    def value(self):
        return dict(ftol=self._ftol_spinbox.value(),
                    xtol=self._xtol_spinbox.value())

    def _set_layout(self):
        _layout = QW.QFormLayout()
        _layout.addRow(BetterQLabel("ftol"), self._ftol_spinbox)
        _layout.addRow(BetterQLabel("xtol"), self._xtol_spinbox)
        self.setLayout(_layout)

    def _set_slot(self):
        self._ftol_spinbox.valueChanged.connect(lambda: self.valueChanged.emit())
        self._xtol_spinbox.valueChanged.connect(lambda: self.valueChanged.emit())


class RangeQWidget(_DefaultQGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Range")
        self.min_spinBox = _DefaultQDoubleSpinBox()
        self.max_spinBox = _DefaultQDoubleSpinBox()

        self._set_layout()
        self._set_init_state()
        self._set_slot()

    def _set_layout(self):
        sub_layout = QW.QFormLayout()
        sub_layout.addRow(_DefaultQLabel('From'), self.min_spinBox)
        sub_layout.addRow(_DefaultQLabel('To'), self.max_spinBox)
        layout = QW.QVBoxLayout()
        layout.addLayout(sub_layout)
        layout.addStretch(1)
        self.setLayout(layout)

    def _set_init_state(self):
        for _box in (self.min_spinBox, self.max_spinBox):
            _box.setRange(200, 900)
            _box.setDecimals(1)
            _box.setSingleStep(0.1)
            _box.setSuffix(' nm')
            _box.setAccelerated(True)
        self.min_spinBox.setValue(200)
        self.max_spinBox.setValue(850)

    def value(self):
        return np.array([self.min_spinBox.value(), self.max_spinBox.value()], dtype=np.float)

    def set_value(self, *, _min, _max):
        if self.isChecked():
            self.min_spinBox.setMaximum(_max)
            self.max_spinBox.setMinimum(_min)
            self.min_spinBox.setValue(_min)
            self.max_spinBox.setValue(_max)
        else:
            return None

    def _set_slot(self):
        def slot_emit():
            self.min_spinBox.setMaximum(self.max_spinBox.value())
            self.max_spinBox.setMinimum(self.min_spinBox.value())
            self.valueChanged.emit()

        self.min_spinBox.valueChanged.connect(slot_emit)
        self.max_spinBox.valueChanged.connect(slot_emit)


class ReportQWidget(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._text_edit = QW.QTextEdit()
        self._text_edit.setReadOnly(True)
        self.setFont(QFont('Consolas', 10))
        layout = QW.QVBoxLayout()
        # layout.addWidget(BetterQLabel('Report'))
        layout.addWidget(self._text_edit)
        self.setLayout(layout)

    def clear_text(self):
        self._text_edit.clear()

    def add_text(self, text):
        self._text_edit.setText(text)


# ----------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    app = QW.QApplication.instance()
    if not app:
        app = QW.QApplication(sys.argv)
    # QW.QApplication.setStyle(QW.QStyleFactory.create('Fusion'))
    app.setStyle(QW.QStyleFactory.create('Fusion'))
    window = QW.QMainWindow()
    cenWidget = QW.QWidget()
    window.setCentralWidget(cenWidget)
    # %%----------------------------------------------------------------------------------------- #
    group = ParaQWidget()
    layout = QW.QHBoxLayout()
    layout.addWidget(group)
    cenWidget.setLayout(layout)
    window.show()
    app.aboutToQuit.connect(app.deleteLater)
