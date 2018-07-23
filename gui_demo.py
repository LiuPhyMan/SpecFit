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
from PyQt5.QtCore import Qt

# class IterableSpectra(QW.QWidget):
#     SPECTRA_LIST = ['OH(A-X)_0-0',
#                     'OH(A-X)_1-0']
#
#     def __init__(self, parent=None):
#         super().__init__(parent)
#
#         self.layout = QW.QVBoxLayout()
#         for _ in self.SPECTRA_LIST:
#             self.layout.addWidget(QW.QCheckBox(_))
#
#         self.setLayout(self.layout)

# !/usr/bin/env python
# coding=utf-8
# from PyQt4.QtGui import *
# from PyQt4.QtCore import *


# class MyDialog(QW.QDialog):
#     def __init__(self, parent=None):
#         super(MyDialog, self).__init__(parent)
#         self.MyTable = QW.QTableWidget(4, 3)
#         self.MyTable.setHorizontalHeaderLabels(['姓名', '身高', '体重'])
#
#         newItem = QW.QTableWidgetItem("松鼠")
#         newItem.setFlags(Qt.ItemIsUserCheckable|Qt.ItemIsEnabled)
#         newItem.setCheckState(Qt.Unchecked)
#         self.MyTable.setItem(0, 0, newItem)
#
#         newItem = QW.QTableWidgetItem("10cm")
#         self.MyTable.setItem(0, 1, newItem)
#
#         newItem = QW.QTableWidgetItem("60g")
#         self.MyTable.setItem(0, 2, newItem)
#
#         layout = QW.QHBoxLayout()
#         layout.addWidget(self.MyTable)
#         self.setLayout(layout)

class MainForm(QW.QDialog):
    def __init__(self, parent=None):

        super(MainForm, self).__init__(parent)
        self.model = QW.BarGraphModel()
        self.barGraphView = QW.BarGraphView()
        self.barGraphView.setModel(self.model)
        self.listView = QW.QListView()
        self.listView.setModel(self.model)
        self.listView.setItemDelegate(QW.BarGraphDelegate(0, 1000, self))
        self.listView.setMaximumWidth(100)
        self.listView.setEditTriggers(QW.QListView.DoubleClicked |
                                      QW.QListView.EditKeyPressed)
        layout = QW.QHBoxLayout()
        layout.addWidget(self.listView)
        layout.addWidget(self.barGraphView, 1)
        self.setLayout(layout)
        self.setWindowTitle("Bar Grapher")






if __name__ == '__main__':
    import sys

    app = QW.QApplication(sys.argv)
    myWindow = MainForm()
    myWindow.show()
    sys.exit(app.exec_())
# cython_temp
#
# app = QW.QApplication.instance()
# if not app:
#     app = QW.QApplication(sys.argv)
# QW.QApplication.setStyle(QW.QStyleFactory.create('Fusion'))
# window = TheWindow()
# window.show()
# app.aboutToQuit.connect(app.deleteLater)
