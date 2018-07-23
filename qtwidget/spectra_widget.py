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
from spectra import OHSpectra, COSpectra, AddSpectra, convolute_to_voigt
from .widgets import (QPlot,
                      BetterButton,
                      BetterQLabel,
                      BetterQDoubleSpinBox,
                      BetterQCheckBox)

_GROUPBOX_TITLE_STYLESHEET = "QGroupBox { font-weight: bold; font-family: UBuntu; font-size: 10pt}"
_DEFAULT_FONT = QFont('Ubuntu', 10, weight=-1)
_DOUBLESPINBOX_STYLESHEET = 'QDoubleSpinBox:disabled {color : rgb(210, 210, 210)}'
_LABEL_STYLESHEET = 'QLabel:disabled {color : rgb(210, 210, 210)}'


class _DefaultQDoubleSpinBox(QW.QDoubleSpinBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont('Ubuntu', 9, italic=True))
        self.setFixedWidth(100)
        self.setAlignment(Qt.AlignRight)


class FWHMQGroupBox(QW.QGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setTitle('FHWM')
        self.setStyleSheet(_GROUPBOX_TITLE_STYLESHEET)

        self.line_shape_combobox = QW.QComboBox()
        self.fwhm_total = BetterQLabel()
        self.fwhm_g_part = _DefaultQDoubleSpinBox()
        self.fwhm_l_part = _DefaultQDoubleSpinBox()
        self._label = dict()
        self._label['Gauss'] = BetterQLabel('Gauss')
        self._label['Loren'] = BetterQLabel('Loren')
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
                    fwhm_g=self.fwhm_g_part.value(),
                    fwhm_l=self.fwhm_l_part.value())

    def set_value(self, *, fwhm_g, fwhm_l):
        self.fwhm_g_part.setValue(fwhm_g)
        self.fwhm_l_part.setValue(fwhm_l)

    def _set_format(self):
        self.line_shape_combobox.setFont(_DEFAULT_FONT)
        self.line_shape_combobox.setCursor(QCursor(Qt.PointingHandCursor))
        self._label['Gauss'].setStyleSheet(_LABEL_STYLESHEET)
        self._label['Loren'].setStyleSheet(_LABEL_STYLESHEET)
        font = QFont('Ubuntu', 13)
        font.setUnderline(True)
        self.fwhm_total.setFont(font)

        for _fwhm in (self.fwhm_g_part, self.fwhm_l_part):
            _fwhm.setStyleSheet(_DOUBLESPINBOX_STYLESHEET)
            _fwhm.setRange(0, 1)
            _fwhm.setDecimals(5)
            _fwhm.setSingleStep(0.0005)
            _fwhm.setAccelerated(True)
            _fwhm.setValue(0.02)

    def _set_layout(self):
        layout = QW.QGridLayout()
        layout.addWidget(self.fwhm_total, 0, 0, 1, 2, alignment=Qt.AlignRight)
        layout.addWidget(self._label['Gauss'], 1, 0, alignment=Qt.AlignRight)
        layout.addWidget(self.fwhm_g_part, 1, 1)
        layout.addWidget(self._label['Loren'], 2, 0, alignment=Qt.AlignRight)
        layout.addWidget(self.fwhm_l_part, 2, 1)
        layout.addWidget(self.line_shape_combobox, 3, 1)
        layout.setRowStretch(4, 1)
        self.setLayout(layout)

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

        if self.para_form() == 'Gaussian':
            set_fwhm_g_state(True)
            set_fwhm_l_state(False)
        if self.para_form() == 'Lorentzian':
            set_fwhm_g_state(False)
            set_fwhm_l_state(True)
        if self.para_form() == 'Voigt':
            set_fwhm_g_state(True)
            set_fwhm_l_state(True)

        self.valueChanged.emit()

    def set_fwhm_total(self):
        if self.para_form() == 'Gaussian':
            _value = self.value()['fwhm_g']
        elif self.para_form() == 'Lorentzian':
            _value = self.value()['fwhm_l']
        elif self.para_form() == 'Voigt':
            _value = convolute_to_voigt(fwhm_G=self.value()['fwhm_g'],
                                        fwhm_L=self.value()['fwhm_l'])
        self.fwhm_total.setText('{i:.5f} nm'.format(i=_value))


class XOffsetQGroupBox(QW.QGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('x_offset')
        self.setCheckable(True)
        self.setStyleSheet(_GROUPBOX_TITLE_STYLESHEET)

        self._combobox = QW.QComboBox()
        self._spinBoxes = dict()
        self._label = dict()
        self._set_combobox()
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
            return lambda x: (x - kwargs['k0'] + kwargs['k1'] * x0) / \
                             (kwargs['k1'] + 1)

    def _set_layout(self):
        layout = QW.QVBoxLayout()
        self.setLayout(layout)
        sub_layout = QW.QFormLayout()
        for _str in ('x0', 'k0', 'k1', 'k2', 'k3'):
            self._label[_str] = BetterQLabel(_str)
            self._label[_str].setStyleSheet(_LABEL_STYLESHEET)
            self._spinBoxes[_str].setStyleSheet(_DOUBLESPINBOX_STYLESHEET)
            sub_layout.addRow(self._label[_str], self._spinBoxes[_str])
        layout.addLayout(sub_layout)
        layout.addWidget(self._combobox)
        layout.addStretch(1)

    def _set_spinBoxes(self):
        for key in ('x0', 'k0', 'k1', 'k2', 'k3'):
            self._spinBoxes[key] = _DefaultQDoubleSpinBox()
            self._spinBoxes[key].setRange(-np.inf, np.inf)
            self._spinBoxes[key].setSingleStep(0.002)
            self._spinBoxes[key].setDecimals(9)
            self._spinBoxes[key].setValue(0)
            self._spinBoxes[key].setAccelerated(True)
        self._spinBoxes['x0'].setSingleStep(1)
        self._spinBoxes['x0'].setDecimals(0)
        self._spinBoxes['x0'].setValue(300)
        self._spinBoxes['k1'].setValue(-0.067)

    def _set_combobox(self):
        self._combobox.setFont(_DEFAULT_FONT)
        self._combobox.setCursor(Qt.PointingHandCursor)
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


class YOffsetQGroupBox(QW.QGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle('y_offset')
        self.setCheckable(True)
        self.setStyleSheet(_GROUPBOX_TITLE_STYLESHEET)

        self.degree_combobox = QW.QComboBox()
        self._spinBoxes = dict()
        self._label = dict(x0=BetterQLabel('x0'),
                           k0=BetterQLabel('k0'), c0=BetterQLabel('c0'),
                           I0=BetterQLabel('I0'))
        self._set_combobox()
        self._set_spinBoxes()
        self._set_layout()
        self._set_slot()
        self.degree_combobox.setCurrentText('linear'.upper())

    def para_form(self):
        return self.degree_combobox.currentText().lower()

    def value(self):
        return dict(para_form=self.para_form(),
                    x0=self._spinBoxes['x0'].value(),
                    k0=self._spinBoxes['k0'].value(),
                    c0=self._spinBoxes['c0'].value(),
                    I0=self._spinBoxes['I0'].value())

    def state(self):
        if self.degree_combobox.currentText().lower() == 'even':
            return [True, False, True, True]
        if self.degree_combobox.currentText().lower() == 'incline':
            return [True, True, True, True]

    def set_value(self, *, x0, k0, c0, I0):
        self._spinBoxes['x0'].setValue(x0)
        self._spinBoxes['k0'].setValue(k0)
        self._spinBoxes['c0'].setValue(c0)
        self._spinBoxes['I0'].setValue(I0)

    def correct_func(self, **kwargs):
        x0, k0, c0, I0 = kwargs['x0'], kwargs['k0'], kwargs['c0'], kwargs['I0']
        return lambda x, y: k0 * (x - x0) + c0 + I0 * y

    def _set_layout(self):
        layout = QW.QVBoxLayout()
        self.setLayout(layout)
        sub_layout = QW.QFormLayout()
        for _str in ('x0', 'k0', 'c0', 'I0'):
            sub_layout.addRow(self._label[_str], self._spinBoxes[_str])
            self._label[_str].setStyleSheet(_LABEL_STYLESHEET)
        layout.addLayout(sub_layout)
        layout.addWidget(self.degree_combobox)
        layout.addStretch(1)

    def _set_spinBoxes(self):
        for key in ('x0', 'k0', 'c0', 'I0'):
            self._spinBoxes[key] = _DefaultQDoubleSpinBox()
            self._spinBoxes[key].setStyleSheet(_DOUBLESPINBOX_STYLESHEET)
            self._spinBoxes[key].setRange(-np.inf, np.inf)
            self._spinBoxes[key].setSingleStep(.01)
            self._spinBoxes[key].setDecimals(4)
            self._spinBoxes[key].setValue(0)
        self._spinBoxes['k0'].setSingleStep(.002)
        self._spinBoxes['x0'].setValue(400)
        self._spinBoxes['x0'].setSingleStep(1)
        self._spinBoxes['I0'].setValue(1)

    def _set_combobox(self):
        self.degree_combobox.setFont(_DEFAULT_FONT)
        self.degree_combobox.setCursor(Qt.PointingHandCursor)
        for key in ['even', 'incline']:
            self.degree_combobox.addItem(key.upper())

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


class GoodnessOfFit(QW.QGroupBox):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Goodness of fit")
        self.setStyleSheet(_GROUPBOX_TITLE_STYLESHEET)
        self._set_labels()
        self._set_layout()

    def _set_layout(self):
        layout = QW.QFormLayout()
        layout.addRow(BetterQLabel('R2'), self._label_r2),
        layout.addRow(BetterQLabel('ChiSquared'), self._label_chisq)
        self.setLayout(layout)

    def _set_labels(self):
        self._label_r2 = BetterQLabel()
        font = QFont('Ubuntu', 13)
        font.setUnderline(True)
        self._label_r2.setFont(font)
        self._label_r2.setText('0.0000')
        self._label_chisq = BetterQLabel()
        self._label_chisq.setFont(font)
        self._label_chisq.setText('0.0000')

    def set_value(self, value):
        self._label.setText("{:.6f}".format(value))


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


class TemperatureQGroupBox(QW.QGroupBox):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setTitle('Temperature')
        self.setStyleSheet(_GROUPBOX_TITLE_STYLESHEET)

        self._vib_input = QW.QDoubleSpinBox()
        self._rot_cold_input = QW.QDoubleSpinBox()
        self._rot_hot_input = QW.QDoubleSpinBox()
        self._hot_ratio = QW.QDoubleSpinBox()
        self._distribution_combobox = QW.QComboBox()
        self._labels = dict()
        self._set_labels()
        self._set_vib()
        self._set_rot()
        self._set_combobox()
        self._set_hot_ratio()
        self._set_layout()
        self._set_format()
        self._set_slot()

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
        if self._distribution_combobox.currentText() == 'one_Trot':
            return [True, True, False, False]
        else:
            return [True, True, True, True]

    def _set_labels(self):
        self._labels['Tvib'] = BetterQLabel('Tvib')
        self._labels['Trot_cold'] = BetterQLabel('Trot_cold')
        self._labels['Trot_hot'] = BetterQLabel('Trot_hot  ')
        self._labels['hot_ratio'] = BetterQLabel('hot_ratio')

    def _set_layout(self):
        _layout = QW.QGridLayout()
        _layout.addWidget(self._labels['Tvib'], 0, 0)
        _layout.addWidget(self._vib_input, 0, 1)
        _layout.addWidget(self._labels['Trot_cold'], 1, 0)
        _layout.addWidget(self._rot_cold_input, 1, 1)
        _layout.addWidget(self._labels['Trot_hot'], 2, 0)
        _layout.addWidget(self._rot_hot_input, 2, 1)
        _layout.addWidget(self._labels['hot_ratio'], 3, 0)
        _layout.addWidget(self._hot_ratio, 3, 1)
        _layout.addWidget(self._distribution_combobox, 4, 1)
        self.setLayout(_layout)

    def _set_format(self):
        for key in ('Tvib', 'Trot_cold', 'Trot_hot', 'hot_ratio'):
            self._labels[key].setStyleSheet(_LABEL_STYLESHEET)
        self._vib_input.setStyleSheet(_DOUBLESPINBOX_STYLESHEET)
        self._rot_cold_input.setStyleSheet(_DOUBLESPINBOX_STYLESHEET)
        self._rot_hot_input.setStyleSheet(_DOUBLESPINBOX_STYLESHEET)
        self._rot_hot_input.setAccelerated(True)
        self._hot_ratio.setStyleSheet(_DOUBLESPINBOX_STYLESHEET)

    def _set_vib(self):
        self._vib_input.setFont(QFont('Ubuntu', 11))
        self._vib_input.setAlignment(Qt.AlignRight)
        self._vib_input.setRange(300, 10000)
        self._vib_input.setSingleStep(10)
        self._vib_input.setValue(3000)
        self._vib_input.setSuffix(' K')
        self._vib_input.setDecimals(0)

    def _set_rot(self):
        for _ in (self._rot_hot_input, self._rot_cold_input):
            _.setFont(QFont('Ubuntu', 11))
            _.setAlignment(Qt.AlignRight)
            _.setRange(300, 20000)
            _.setSingleStep(10)
            _.setValue(3000)
            _.setSuffix(' K')
            _.setDecimals(0)
        self._rot_hot_input.setSingleStep(50)

    def _set_hot_ratio(self):
        self._hot_ratio.setFont(QFont('Ubuntu', 11))
        self._hot_ratio.setAlignment(Qt.AlignRight)
        self._hot_ratio.setRange(0, 1)
        self._hot_ratio.setSingleStep(0.05)
        self._hot_ratio.setValue(0.5)
        self._hot_ratio.setDecimals(2)

    def _set_combobox(self):
        self._distribution_combobox.setFont(_DEFAULT_FONT)
        self._distribution_combobox.setCursor(Qt.PointingHandCursor)
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
            if self._distribution_combobox.currentText() == 'one_Trot':
                self._labels['Trot_cold'].setText('Trot')
            elif self._distribution_combobox.currentText() == 'two_Trot':
                self._labels['Trot_cold'].setText('Trot_cold')
            slot_emit()

        self._vib_input.valueChanged.connect(slot_emit)
        self._rot_cold_input.valueChanged.connect(slot_emit)
        self._rot_hot_input.valueChanged.connect(slot_emit)
        self._hot_ratio.valueChanged.connect(slot_emit)
        self._distribution_combobox.currentTextChanged.connect(_combobox_callback)
        _combobox_callback()


class ParaQWidget(QW.QWidget):
    valueChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QW.QGridLayout()
        self._temperature = TemperatureQGroupBox()
        self._fwhm = FWHMQGroupBox()
        self._x_offset = XOffsetQGroupBox()
        self._y_offset = YOffsetQGroupBox()

        self.layout.addWidget(self._temperature, 0, 0)
        self.layout.addWidget(self._fwhm, 0, 1)
        self.layout.addWidget(self._x_offset, 0, 2)
        self.layout.addWidget(self._y_offset, 0, 3)

        self.setLayout(self.layout)
        self._set_slot()

    def state(self):
        return dict(temperature=self._temperature.isChecked(),
                    fwhm=self._fwhm.isChecked(),
                    x_offset=self._x_offset.isChecked(),
                    y_offset=self._y_offset.isChecked())

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
        return dict(temperature=self._temperature.value(),
                    fwhm=self._fwhm.value(),
                    x_offset=self._x_offset.value(),
                    y_offset=self._y_offset.value())

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

    def _set_slot(self):
        def slot_emit():
            self.valueChanged.emit()

        self._temperature.valueChanged.connect(slot_emit)
        self._fwhm.valueChanged.connect(slot_emit)
        self._x_offset.valueChanged.connect(slot_emit)
        self._y_offset.valueChanged.connect(slot_emit)


class SpectraPlot(QPlot):
    _LABEL_FONTDICT = {'family': 'Consolas', 'size': 14}
    _TEXT_FONTDICT = {'family': 'Consolas', 'size': 11}

    def __init__(self, parent=None, width=12, height=5):
        super().__init__(parent, figsize=(width, height), dpi=100,
                         toolbar_position='left')
        self.toolbar.setIconSize(QSize(24, 24))
        self.axes = self.figure.add_subplot(111)
        # self.axes.xaxis.set_major_locator(ticker.MultipleLocator(5.00))
        # self.axes.xaxis.set_minor_locator(ticker.MultipleLocator(1.00))
        # set ticker
        locator = ticker.MaxNLocator(nbins=20)
        self.axes.xaxis.set_major_locator(locator)
        self.axes.set_xlabel('wavelength [nm]', fontdict=self._LABEL_FONTDICT)
        self.axes.set_ylabel('intensity [a.u.]', fontdict=self._LABEL_FONTDICT)
        self.axes.set_xlim(250, 850)
        self.axes.set_ylim(-0.2, 1)

        self.axes.grid()
        self.sim_line, = self.axes.plot([], [], 'r-', linewidth=.8)
        # self.exp_line, = self.axes.plot([], [], 'bo-', linewidth=.5, markersize=.5)
        self.exp_lines = []
        self._branch_lines = []
        self._texts = []
        self.figure.tight_layout()

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
    _DICT = {'OH(A-X)': ['0-0 308.9 nm',
                         '1-1 314.5 nm',
                         '2-2 320.7 nm',
                         '1-0 282.8 nm',
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
                         '1-2 469.7 nm']}

    def __init__(self, parent=None):
        super().__init__(parent)
        self.spectra_func = None
        self._set_node_check('OH(A-X)')
        self.item_changed_callback()

    def item_changed_callback(self):
        spectra_list = self.get_select_state()

        def get_spectra(_spectra):
            temp = re.fullmatch(r"([^()]+)\(([^()]+)\)_(\d+)-(\d+)", _spectra).groups()
            molecule, band, v_upper_str, v_lower_str = temp
            if molecule == 'OH':
                func = OHSpectra
            if molecule == 'CO':
                func = COSpectra
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
                    selected_list.append('{spec}_{band}'.format(spec=key, band=_.text(0)[:3]))
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
        _layout = QW.QFormLayout()
        self._width_spinbox = QW.QSpinBox()
        self._height_spinbox = QW.QSpinBox()
        self._width_spinbox.setValue(init_width)
        self._height_spinbox.setValue(init_height)
        _layout.addRow('Width', self._width_spinbox)
        _layout.addRow('Height', self._height_spinbox)
        self.setLayout(_layout)
        self._set_slot()

    def value(self):
        return self._width_spinbox.value(), self._height_spinbox.value()

    def _set_slot(self):
        self._width_spinbox.valueChanged.connect(lambda: self.valueChanged.emit())
        self._height_spinbox.valueChanged.connect(lambda: self.valueChanged.emit())


class RangeQWidget(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.min_spinBox = BetterQDoubleSpinBox()
        self.max_spinBox = BetterQDoubleSpinBox()

        sub_layout = QW.QFormLayout()
        sub_layout.addRow(BetterQLabel('From'), self.min_spinBox)
        sub_layout.addRow(BetterQLabel('To'), self.max_spinBox)
        layout = QW.QVBoxLayout()
        layout.addLayout(sub_layout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.set_init_state()

    def set_init_state(self):
        self.min_spinBox.setRange(200, 900)
        self.min_spinBox.setDecimals(1)
        self.min_spinBox.setSingleStep(0.1)
        self.min_spinBox.setFont(QFont('Ubuntu', 13))
        self.min_spinBox.setSuffix(' nm')
        self.min_spinBox.setAlignment(Qt.AlignRight)
        self.min_spinBox.setAccelerated(True)
        self.min_spinBox.setValue(200)

        self.max_spinBox.setRange(200, 900)
        self.max_spinBox.setDecimals(1)
        self.max_spinBox.setSingleStep(0.1)
        self.max_spinBox.setFont(QFont('Ubuntu', 13))
        self.max_spinBox.setSuffix(' nm')
        self.max_spinBox.setAlignment(Qt.AlignRight)
        self.max_spinBox.setAccelerated(True)
        self.max_spinBox.setValue(850)

    def value(self):
        return np.array([self.min_spinBox.value(), self.max_spinBox.value()], dtype=np.float)


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
        self.setFixedWidth(450)

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
