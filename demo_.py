#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15:36 2018/4/10

@author:    Liu Jinbao
@mail:      liu.jinbao@outlook.com
@project:   test
@IDE:       PyCharm
"""
import sys
from PyQt5 import QtWidgets as QW


class IterableSpectra(QW.QWidget):
    SPECTRA_LIST = ['OH(A-X)_0-0',
                    'OH(A-X)_1-0']

    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QW.QVBoxLayout()
        for _ in self.SPECTRA_LIST:
            self.layout.addWidget(QW.QCheckBox(_))

        self.setLayout(self.layout)


class TheWindow(QW.QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(800, 600)
        self.cenWidget = QW.QWidget()
        self.setCentralWidget(self.cenWidget)

        layout = QW.QVBoxLayout()
        # layout.addWidget(IterableSpectra())
        self.cenWidget.setLayout(layout)


app = QW.QApplication.instance()
if not app:
    app = QW.QApplication(sys.argv)
QW.QApplication.setStyle(QW.QStyleFactory.create('Fusion'))
window = TheWindow()
window.show()
app.aboutToQuit.connect(app.deleteLater)
