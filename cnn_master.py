
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow
from PyQt5.QtCore import QMimeData
from cnn_predict import Predictor
from cnn_gui import Ui_MainWindow
import folder_inspector

class CnnMain(QMainWindow, Ui_MainWindow):

    def __init__(self, savedModel, labelDict, parent=None):

        # Set up Interface #
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.predictor = Predictor(labelDict,savedModel)

    def dragEnterEvent(self, e):
        print("image dragged")
        if e.mimeData().hasUrls():
            print("image recognised")
            e.accept()
        else:
            print("this is not an image")
            e.ignore()

    def dropEvent(self, e):
        print("image dropped")
        name = e.mimeData().text()[7:].split("/")[-1]
        print(name)
        print(e.mimeData().text()[7:])
        prediction = self.predictor.predict(e.mimeData().text()[7:])

        print(prediction)

        self.leName.setText(name)
        self.lePrediction.setText(prediction)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    MainWindow = CnnMain("first_try.h5", folder_inspector.getClassDict("caribe_train"))
    MainWindow.show()
    sys.exit(app.exec_())