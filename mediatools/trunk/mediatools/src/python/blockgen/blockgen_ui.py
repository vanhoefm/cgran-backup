# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'blockgen_ui.ui'
#
# Created: Wed May 18 23:27:06 2011
#      by: PyQt4 UI code generator 4.7.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_Blockgen(object):
    def setupUi(self, Blockgen):
        Blockgen.setObjectName("Blockgen")
        Blockgen.resize(1001, 753)
        self.centralwidget = QtGui.QWidget(Blockgen)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtGui.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_3 = QtGui.QLabel(self.centralwidget)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 4, 0, 1, 1)
        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 3, 0, 1, 1)
        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 2, 0, 1, 1)
        self.label_16 = QtGui.QLabel(self.centralwidget)
        self.label_16.setObjectName("label_16")
        self.gridLayout.addWidget(self.label_16, 0, 0, 1, 1)
        self.selectSuper = QtGui.QComboBox(self.centralwidget)
        self.selectSuper.setObjectName("selectSuper")
        self.selectSuper.addItem("")
        self.selectSuper.addItem("")
        self.gridLayout.addWidget(self.selectSuper, 4, 1, 1, 2)
        self.txtBlock = QtGui.QLineEdit(self.centralwidget)
        self.txtBlock.setObjectName("txtBlock")
        self.gridLayout.addWidget(self.txtBlock, 3, 1, 1, 2)
        self.txtModule = QtGui.QLineEdit(self.centralwidget)
        self.txtModule.setObjectName("txtModule")
        self.gridLayout.addWidget(self.txtModule, 2, 1, 1, 2)
        self.txtModDir = QtGui.QLineEdit(self.centralwidget)
        self.txtModDir.setObjectName("txtModDir")
        self.gridLayout.addWidget(self.txtModDir, 0, 1, 1, 1)
        self.txtModDirBrowse = QtGui.QPushButton(self.centralwidget)
        self.txtModDirBrowse.setObjectName("txtModDirBrowse")
        self.gridLayout.addWidget(self.txtModDirBrowse, 0, 2, 1, 1)
        self.label_10 = QtGui.QLabel(self.centralwidget)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 0, 3, 1, 1)
        self.label_11 = QtGui.QLabel(self.centralwidget)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 2, 3, 1, 1)
        self.label_12 = QtGui.QLabel(self.centralwidget)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 3, 3, 1, 1)
        self.label_17 = QtGui.QLabel(self.centralwidget)
        self.label_17.setObjectName("label_17")
        self.gridLayout.addWidget(self.label_17, 4, 3, 1, 1)
        self.txtFnX = QtGui.QLineEdit(self.centralwidget)
        self.txtFnX.setEnabled(False)
        self.txtFnX.setReadOnly(False)
        self.txtFnX.setObjectName("txtFnX")
        self.gridLayout.addWidget(self.txtFnX, 4, 4, 1, 1)
        self.txtFnI = QtGui.QLineEdit(self.centralwidget)
        self.txtFnI.setEnabled(False)
        self.txtFnI.setObjectName("txtFnI")
        self.gridLayout.addWidget(self.txtFnI, 3, 4, 1, 1)
        self.txtFnH = QtGui.QLineEdit(self.centralwidget)
        self.txtFnH.setEnabled(False)
        self.txtFnH.setObjectName("txtFnH")
        self.gridLayout.addWidget(self.txtFnH, 2, 4, 1, 1)
        self.txtFnCC = QtGui.QLineEdit(self.centralwidget)
        self.txtFnCC.setEnabled(False)
        self.txtFnCC.setObjectName("txtFnCC")
        self.gridLayout.addWidget(self.txtFnCC, 0, 4, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.tabWidget = QtGui.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtGui.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_4 = QtGui.QGridLayout(self.tab)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_4 = QtGui.QLabel(self.tab)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 0, 0, 1, 4)
        self.label_5 = QtGui.QLabel(self.tab)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 0, 4, 1, 4)
        self.addInSig = QtGui.QPushButton(self.tab)
        self.addInSig.setObjectName("addInSig")
        self.gridLayout_3.addWidget(self.addInSig, 1, 0, 1, 1)
        self.selectInSig = QtGui.QComboBox(self.tab)
        self.selectInSig.setObjectName("selectInSig")
        self.gridLayout_3.addWidget(self.selectInSig, 1, 1, 1, 3)
        self.addOutSig = QtGui.QPushButton(self.tab)
        self.addOutSig.setObjectName("addOutSig")
        self.gridLayout_3.addWidget(self.addOutSig, 1, 4, 1, 1)
        self.inSigTable = QtGui.QTableView(self.tab)
        self.inSigTable.setDragEnabled(True)
        self.inSigTable.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.inSigTable.setObjectName("inSigTable")
        self.gridLayout_3.addWidget(self.inSigTable, 2, 0, 1, 4)
        self.outSigTable = QtGui.QTableView(self.tab)
        self.outSigTable.setDragEnabled(True)
        self.outSigTable.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.outSigTable.setObjectName("outSigTable")
        self.gridLayout_3.addWidget(self.outSigTable, 2, 4, 1, 4)
        self.label_6 = QtGui.QLabel(self.tab)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 3, 0, 1, 1)
        self.txtInMin = QtGui.QLineEdit(self.tab)
        self.txtInMin.setObjectName("txtInMin")
        self.gridLayout_3.addWidget(self.txtInMin, 3, 1, 1, 1)
        self.label_7 = QtGui.QLabel(self.tab)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 3, 2, 1, 1)
        self.txtInMax = QtGui.QLineEdit(self.tab)
        self.txtInMax.setObjectName("txtInMax")
        self.gridLayout_3.addWidget(self.txtInMax, 3, 3, 1, 1)
        self.label_8 = QtGui.QLabel(self.tab)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 3, 4, 1, 1)
        self.txtOutMin = QtGui.QLineEdit(self.tab)
        self.txtOutMin.setObjectName("txtOutMin")
        self.gridLayout_3.addWidget(self.txtOutMin, 3, 5, 1, 1)
        self.label_9 = QtGui.QLabel(self.tab)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 3, 6, 1, 1)
        self.txtOutMax = QtGui.QLineEdit(self.tab)
        self.txtOutMax.setObjectName("txtOutMax")
        self.gridLayout_3.addWidget(self.txtOutMax, 3, 7, 1, 1)
        self.selectOutSig = QtGui.QComboBox(self.tab)
        self.selectOutSig.setObjectName("selectOutSig")
        self.gridLayout_3.addWidget(self.selectOutSig, 1, 5, 1, 3)
        self.gridLayout_4.addLayout(self.gridLayout_3, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtGui.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_6 = QtGui.QGridLayout(self.tab_2)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.gridLayout_5 = QtGui.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.methodTable = QtGui.QTableView(self.tab_2)
        self.methodTable.setEnabled(True)
        self.methodTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.methodTable.setTabKeyNavigation(True)
        self.methodTable.setDragEnabled(True)
        self.methodTable.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.methodTable.setObjectName("methodTable")
        self.methodTable.horizontalHeader().setVisible(False)
        self.methodTable.horizontalHeader().setStretchLastSection(True)
        self.methodTable.verticalHeader().setVisible(False)
        self.gridLayout_5.addWidget(self.methodTable, 2, 0, 1, 2)
        self.argTypeTable = QtGui.QTableView(self.tab_2)
        self.argTypeTable.setEnabled(True)
        self.argTypeTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.argTypeTable.setDragEnabled(True)
        self.argTypeTable.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.argTypeTable.setObjectName("argTypeTable")
        self.argTypeTable.horizontalHeader().setVisible(False)
        self.argTypeTable.horizontalHeader().setStretchLastSection(True)
        self.argTypeTable.verticalHeader().setVisible(False)
        self.gridLayout_5.addWidget(self.argTypeTable, 2, 2, 1, 2)
        self.argNameTable = QtGui.QTableView(self.tab_2)
        self.argNameTable.setEnabled(True)
        self.argNameTable.setFrameShape(QtGui.QFrame.StyledPanel)
        self.argNameTable.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.argNameTable.setDragEnabled(True)
        self.argNameTable.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        self.argNameTable.setObjectName("argNameTable")
        self.argNameTable.horizontalHeader().setVisible(False)
        self.argNameTable.horizontalHeader().setStretchLastSection(True)
        self.argNameTable.verticalHeader().setVisible(False)
        self.gridLayout_5.addWidget(self.argNameTable, 2, 4, 1, 1)
        self.addMethod = QtGui.QPushButton(self.tab_2)
        self.addMethod.setObjectName("addMethod")
        self.gridLayout_5.addWidget(self.addMethod, 1, 0, 1, 1)
        self.addMethodName = QtGui.QLineEdit(self.tab_2)
        self.addMethodName.setObjectName("addMethodName")
        self.gridLayout_5.addWidget(self.addMethodName, 1, 1, 1, 1)
        self.addArg = QtGui.QPushButton(self.tab_2)
        self.addArg.setObjectName("addArg")
        self.gridLayout_5.addWidget(self.addArg, 1, 2, 1, 1)
        self.selectArgType = QtGui.QComboBox(self.tab_2)
        self.selectArgType.setObjectName("selectArgType")
        self.gridLayout_5.addWidget(self.selectArgType, 1, 3, 1, 1)
        self.argName = QtGui.QLineEdit(self.tab_2)
        self.argName.setObjectName("argName")
        self.gridLayout_5.addWidget(self.argName, 1, 4, 1, 1)
        self.label_15 = QtGui.QLabel(self.tab_2)
        self.label_15.setObjectName("label_15")
        self.gridLayout_5.addWidget(self.label_15, 0, 0, 1, 2)
        self.label_13 = QtGui.QLabel(self.tab_2)
        self.label_13.setObjectName("label_13")
        self.gridLayout_5.addWidget(self.label_13, 0, 2, 1, 3)
        self.label_14 = QtGui.QLabel(self.tab_2)
        self.label_14.setObjectName("label_14")
        self.gridLayout_5.addWidget(self.label_14, 3, 0, 1, 1)
        self.selectMethodReturnType = QtGui.QComboBox(self.tab_2)
        self.selectMethodReturnType.setObjectName("selectMethodReturnType")
        self.gridLayout_5.addWidget(self.selectMethodReturnType, 3, 1, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 0, 1, 1)
        self.tabWidget.addTab(self.tab_2, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.genButton = QtGui.QPushButton(self.centralwidget)
        self.genButton.setObjectName("genButton")
        self.verticalLayout.addWidget(self.genButton)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        Blockgen.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(Blockgen)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1001, 25))
        self.menubar.setObjectName("menubar")
        Blockgen.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(Blockgen)
        self.statusbar.setObjectName("statusbar")
        Blockgen.setStatusBar(self.statusbar)

        self.retranslateUi(Blockgen)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Blockgen)

    def retranslateUi(self, Blockgen):
        Blockgen.setWindowTitle(QtGui.QApplication.translate("Blockgen", "GR Rapid BlockGen", None, QtGui.QApplication.UnicodeUTF8))
        self.label_3.setText(QtGui.QApplication.translate("Blockgen", "Superclass", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Blockgen", "Block Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Blockgen", "Module Name", None, QtGui.QApplication.UnicodeUTF8))
        self.label_16.setText(QtGui.QApplication.translate("Blockgen", "Module Directory", None, QtGui.QApplication.UnicodeUTF8))
        self.selectSuper.setItemText(0, QtGui.QApplication.translate("Blockgen", "gr_sync_block", None, QtGui.QApplication.UnicodeUTF8))
        self.selectSuper.setItemText(1, QtGui.QApplication.translate("Blockgen", "gr_block", None, QtGui.QApplication.UnicodeUTF8))
        self.txtBlock.setText(QtGui.QApplication.translate("Blockgen", "example", None, QtGui.QApplication.UnicodeUTF8))
        self.txtModule.setText(QtGui.QApplication.translate("Blockgen", "mediatools", None, QtGui.QApplication.UnicodeUTF8))
        self.txtModDir.setText(QtGui.QApplication.translate("Blockgen", ".", None, QtGui.QApplication.UnicodeUTF8))
        self.txtModDirBrowse.setText(QtGui.QApplication.translate("Blockgen", "Browse", None, QtGui.QApplication.UnicodeUTF8))
        self.label_10.setText(QtGui.QApplication.translate("Blockgen", "CC: ", None, QtGui.QApplication.UnicodeUTF8))
        self.label_11.setText(QtGui.QApplication.translate("Blockgen", "HH: ", None, QtGui.QApplication.UnicodeUTF8))
        self.label_12.setText(QtGui.QApplication.translate("Blockgen", "I:", None, QtGui.QApplication.UnicodeUTF8))
        self.label_17.setText(QtGui.QApplication.translate("Blockgen", "XML:", None, QtGui.QApplication.UnicodeUTF8))
        self.txtFnX.setText(QtGui.QApplication.translate("Blockgen", "mediatools_example.xml", None, QtGui.QApplication.UnicodeUTF8))
        self.txtFnI.setText(QtGui.QApplication.translate("Blockgen", "mediatools_example.i", None, QtGui.QApplication.UnicodeUTF8))
        self.txtFnH.setText(QtGui.QApplication.translate("Blockgen", "mediatools_example.hh", None, QtGui.QApplication.UnicodeUTF8))
        self.txtFnCC.setText(QtGui.QApplication.translate("Blockgen", "mediatools_example.cc", None, QtGui.QApplication.UnicodeUTF8))
        self.label_4.setText(QtGui.QApplication.translate("Blockgen", "Input Signatures", None, QtGui.QApplication.UnicodeUTF8))
        self.label_5.setText(QtGui.QApplication.translate("Blockgen", "Output Signatures", None, QtGui.QApplication.UnicodeUTF8))
        self.addInSig.setText(QtGui.QApplication.translate("Blockgen", "Add", None, QtGui.QApplication.UnicodeUTF8))
        self.addOutSig.setText(QtGui.QApplication.translate("Blockgen", "Add", None, QtGui.QApplication.UnicodeUTF8))
        self.label_6.setText(QtGui.QApplication.translate("Blockgen", "In: Min", None, QtGui.QApplication.UnicodeUTF8))
        self.txtInMin.setText(QtGui.QApplication.translate("Blockgen", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_7.setText(QtGui.QApplication.translate("Blockgen", "Max", None, QtGui.QApplication.UnicodeUTF8))
        self.txtInMax.setText(QtGui.QApplication.translate("Blockgen", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_8.setText(QtGui.QApplication.translate("Blockgen", "Out: Min", None, QtGui.QApplication.UnicodeUTF8))
        self.txtOutMin.setText(QtGui.QApplication.translate("Blockgen", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.label_9.setText(QtGui.QApplication.translate("Blockgen", "Max", None, QtGui.QApplication.UnicodeUTF8))
        self.txtOutMax.setText(QtGui.QApplication.translate("Blockgen", "0", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), QtGui.QApplication.translate("Blockgen", "I/O Signatures", None, QtGui.QApplication.UnicodeUTF8))
        self.addMethod.setText(QtGui.QApplication.translate("Blockgen", "Add Method", None, QtGui.QApplication.UnicodeUTF8))
        self.addMethodName.setText(QtGui.QApplication.translate("Blockgen", "new_method", None, QtGui.QApplication.UnicodeUTF8))
        self.addArg.setText(QtGui.QApplication.translate("Blockgen", "Add Arg", None, QtGui.QApplication.UnicodeUTF8))
        self.label_15.setText(QtGui.QApplication.translate("Blockgen", "Method List", None, QtGui.QApplication.UnicodeUTF8))
        self.label_13.setText(QtGui.QApplication.translate("Blockgen", "Selected Method Argument Type/Names", None, QtGui.QApplication.UnicodeUTF8))
        self.label_14.setText(QtGui.QApplication.translate("Blockgen", "Return Type: ", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), QtGui.QApplication.translate("Blockgen", "Method Definition", None, QtGui.QApplication.UnicodeUTF8))
        self.genButton.setText(QtGui.QApplication.translate("Blockgen", "Generate", None, QtGui.QApplication.UnicodeUTF8))

