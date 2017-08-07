# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'cnn_gui.ui'
#
# Created by: PyQt5 UI code generator 5.8.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setAcceptDrops(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(290, 10, 461, 85))
        self.widget.setObjectName("widget")
        self.formLayout = QtWidgets.QFormLayout(self.widget)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setObjectName("label")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label)
        self.leName = QtWidgets.QLineEdit(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.leName.sizePolicy().hasHeightForWidth())
        self.leName.setSizePolicy(sizePolicy)
        self.leName.setObjectName("leName")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.leName)
        self.lePrediction = QtWidgets.QLineEdit(self.widget)
        self.lePrediction.setObjectName("lePrediction")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lePrediction)
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setObjectName("label_3")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_3)
        self.leProb = QtWidgets.QLineEdit(self.widget)
        self.leProb.setObjectName("leProb")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.leProb)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "CNN"))
        self.label.setText(_translate("MainWindow", "Image Name"))
        self.label_2.setText(_translate("MainWindow", "Prediction"))
        self.label_3.setText(_translate("MainWindow", "Probability"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

