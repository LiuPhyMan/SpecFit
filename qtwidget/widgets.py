#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 7:40 2018/3/30

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   PlasmaChemistry
@IDE:       PyCharm
"""
import math
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter
# from PyQt5 import QtWidgets as QW
# from PyQt5.QtCore import Qt, QSize
# from PyQt5.QtGui import QCursor, QFont
from PySide6 import QtWidgets as QW
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QCursor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar


class LogScaleSlier(QW.QSlider):

    def __init__(self, parent):
        super().__init__(parent)
        self.setMaximum(500)
        self.setMinimum(0)
        self.setTickInterval(100)
        self.setTickPosition(QW.QSlider.TicksAbove)
        self.setSingleStep(1)
        self.setPageStep(1)
        self.setOrientation(Qt.Horizontal)
        self.setCursor(QCursor(Qt.PointingHandCursor))

    def set_range(self, time_seq):
        log_t1 = math.log10(time_seq[1])
        log_tn = math.log10(time_seq[-2])
        self.value_seq = np.hstack((time_seq[0],
                                    np.logspace(log_t1, log_tn, num=499),
                                    time_seq[-1]))

    def value(self):
        index = QW.QSlider.value(self)
        return self.value_seq[index]


class PlotCanvas(FigureCanvas):

    def __init__(self, parent, _figure):
        FigureCanvas.__init__(self, _figure)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QW.QSizePolicy.Expanding, QW.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class QPlot(QW.QWidget):
    def __init__(self, parent=None, figsize=(5, 4), dpi=100, toolbar_position='left'):
        super().__init__(parent)
        assert toolbar_position in ('left', 'right', 'top', 'bottom'), toolbar_position
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.canvas = PlotCanvas(parent, self.figure)
        self.canvas.setFixedSize(figsize[0] * dpi, figsize[1] * dpi)
        self.toolbar = NavigationToolbar(self.canvas, parent=parent, coordinates=False)
        self.toolbar.setIconSize(QSize(16, 16))
        self.toolbar.update()
        if toolbar_position in ('left', 'right'):
            self.toolbar.setOrientation(Qt.Vertical)
            layout = QW.QHBoxLayout(parent)
        if toolbar_position in ('bottom', 'top'):
            self.toolbar.setOrientation(Qt.Horizontal)
            layout = QW.QVBoxLayout(parent)
        if toolbar_position in ('left', 'top'):
            layout.addWidget(self.toolbar)
            layout.addWidget(self.canvas)
        if toolbar_position in ('right', 'bottom'):
            layout.addWidget(self.canvas)
            layout.addWidget(self.toolbar)
        layout.addStretch(1)
        self.setLayout(layout)


class EEDFQPlot(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.qplot = QPlot(parent)
        self.axes = self.qplot.figure.add_subplot(111)
        self.time_list = QW.QListWidget(parent)
        self.time_list.setMaximumWidth(90)
        self.time_list.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
        self.data_plot = dict()
        layout = QW.QHBoxLayout()
        layout.addWidget(self.qplot)
        layout.addWidget(self.time_list)
        self.setLayout(layout)
        self.initialize()

    def initialize(self):
        self.axes.clear()
        self.axes.grid()
        self.axes.set_yscale('log')
        self.axes.set_xlabel('Energy [eV]')
        self.axes.set_ylabel(r'EEPF [$eV^{-3/2}$]')
        self.axes.set_xlim(0, 30)
        self.axes.set_ylim(1e-10, 1e1)
        self.axes.yaxis.set_major_formatter(FormatStrFormatter('%.0e'))
        self.axes.set_position([0.15, 0.15, 0.7, 0.8])
        self.axes.tick_params(axis='both', labelsize=8)
        # self.axes.set_yticks([10 ** _ for _ in range(-10, 2)])

    def time_list_selected_index(self):
        return [_.row() for _ in self.time_list.selectionModel().selectedRows()]

    def canvas_draw(self):
        self.qplot.canvas.draw()

    def import_data(self, *, energy_points, time_seq, eepf_seq):
        self.data_plot['energy_points'] = energy_points
        self.data_plot['time_seq'] = time_seq
        self.data_plot['eepf_seq'] = eepf_seq
        for _ in time_seq:
            self.time_list.addItem('{t:.2e}'.format(t=_))

    def plot_selected(self):
        print('plot_selected')
        index = self.time_list_selected_index()
        self.plot(xdata=self.data_plot['energy_points'],
                  ydata=self.data_plot['eepf_seq'][index],
                  time_labels=self.data_plot['time_seq'][index])

    def clear_plot(self):
        while len(self.axes.lines):
            self.axes.lines.pop(0)
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.axes.set_prop_cycle('color', color_list)
        self.canvas_draw()

    def plot(self, *, xdata, ydata, time_labels):
        self.clear_plot()
        assert ydata.ndim == 2, 'The ndim of ydata should not be {}.'.format(ydata.ndim)
        labels = ['time:{:.2e} s'.format(_) for _ in time_labels]
        for _y, _label in zip(ydata, labels):
            self.axes.plot(xdata, _y, linewidth=1, marker='.', markersize=2, label=_label)
        self.axes.legend(fontsize='x-small')
        self.canvas_draw()


class DensityPlot(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.qplot = QPlot(parent)
        self.axes = self.qplot.figure.add_subplot(111)
        self.specie_list = QW.QListWidget(parent)
        self.specie_list.setFixedWidth(90)
        self.specie_list.setSelectionMode(QW.QAbstractItemView.ExtendedSelection)
        self.data_plot = dict()
        layout = QW.QHBoxLayout()
        layout.addWidget(self.qplot)
        layout.addWidget(self.specie_list)
        self.setLayout(layout)
        self.initialize()

    def initialize(self):
        self.axes.clear()
        self.axes.grid()
        self.axes.set_xscale('log')
        self.axes.set_yscale('log')
        self.axes.set_xlabel('Time [s]')
        self.axes.set_ylabel('Density')
        self.axes.set_position([.15, .15, .7, .8])
        self.vertical_line = self.axes.axvline(x=-np.inf, alpha=.5)
        self.vertical_line.set_xdata(-np.inf)

    def specie_list_selected_index(self):
        return [_.row() for _ in self.specie_list.selectionModel().selectedRows()]

    def canvas_draw(self):
        self.qplot.canvas.draw()

    def import_data(self, *, time_seq, density_seq, species):
        self.data_plot['time_seq'] = time_seq
        self.data_plot['density_seq'] = density_seq
        self.data_plot['species'] = species
        for _ in self.data_plot['species']:
            self.specie_list.addItem(_)

    def plot_selected(self):
        index = self.specie_list_selected_index()
        self.plot(xdata=self.data_plot['time_seq'],
                  ydata=self.data_plot['density_seq'][index],
                  density_labels=self.data_plot['species'][index])

    def clear_plot(self):
        x_vline = self.vertical_line.get_xydata()[0, 0]
        while len(self.axes.lines):
            self.axes.lines.pop(0)
        color_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        self.axes.set_prop_cycle('color', color_list)
        self.vertical_line = self.axes.axvline(x=x_vline, alpha=.5)
        self.canvas_draw()

    def plot(self, *, xdata, ydata, density_labels):
        self.clear_plot()
        assert ydata.ndim == 2
        for _y, _label in zip(ydata, density_labels):
            self.axes.plot(xdata, _y, linewidth=.5, marker='.', markersize=2, label=_label)
        self.axes.legend(fontsize='x-small')
        self.canvas_draw()

    def plot_vline(self, *, x):
        self.vertical_line.set_xdata(x)
        self.canvas_draw()


class NeTeQPlot(QW.QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.qplot = QPlot(parent)
        self.axes_Te = self.qplot.figure.add_subplot(111)
        self.axes_ne = self.axes_Te.twinx()
        self.data_plot = dict()
        layout = QW.QHBoxLayout()
        layout.addWidget(self.qplot)
        self.setLayout(layout)
        self.initialize()

    def initialize(self):
        self.axes_Te.clear()
        self.axes_ne.clear()
        self.axes_ne.set_yscale('log')
        self.axes_ne.set_ylabel('ne [m^3]')
        self.axes_Te.set_xscale('log')
        self.axes_Te.set_xlabel('Time [s]')
        self.axes_Te.set_ylabel('Te [eV]')
        self.axes_ne.set_position([0.15, 0.15, 0.7, 0.8])
        self.axes_Te.set_position([0.15, 0.15, 0.7, 0.8])
        for tick in self.axes_Te.get_yticklabels():
            tick.set_color('r')
        for tick in self.axes_ne.get_yticklabels():
            tick.set_color('b')
        self.vertical_line = self.axes_Te.axvline(x=-np.inf, alpha=0.5)

    def canvas_draw(self):
        self.qplot.canvas.draw()

    def import_data(self, *, time_seq, ne_seq, Te_seq):
        self.data_plot['time_seq'] = time_seq
        self.data_plot['ne_seq'] = ne_seq
        self.data_plot['Te_seq'] = Te_seq
        self.initialize()
        self.plot(time_seq=time_seq, ne_seq=ne_seq, Te_seq=Te_seq)

    def clear_plot(self):
        while len(self.axes_Te.lines):
            self.axes_Te.lines.pop(0)
        self.canvas_draw()

    def plot(self, *, time_seq, ne_seq, Te_seq):
        self.axes_Te.plot(time_seq, Te_seq, linewidth=.5, color='r', marker=',', markersize=2)
        self.axes_ne.set_ylim(10 ** (np.floor(np.log10(ne_seq.min())) - 0.1),
                              10 ** (np.ceil(np.log10(ne_seq.max())) + 0.1))
        self.axes_ne.plot(time_seq, ne_seq, linewidth=0.5, color='b', marker='.', markersize=2)
        self.canvas_draw()

    def plot_vline(self, *, x):
        self.vertical_line.set_xdata(x)
        self.canvas_draw()
