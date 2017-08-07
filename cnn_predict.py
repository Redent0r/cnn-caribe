from PIL import Image
from keras.models import load_model
import numpy as np

class Predictor:

	def __init__(self, label_dict, model_loc):
		print("loading model")
		self.model = load_model('first_try.h5')
		self.label_dict = label_dict
		print("model succesfully loaded")

	def predict(self, img_location):

		# import images
		img = Image.open(img_location).resize((150, 150))
		#img = img.resize((150, 150))
		x = np.asarray(img).reshape((1, 150, 150, 3))
		#x = x.reshape((1, 150, 150, 3))
		return self.label_dict[self.model.predict_classes(x)[0]]

# # y labels
# labels = {0:"fish_15", 1:"fish_17", 2:"fish_18", 3:"fish_22"}
# pred = Predictor(labels, "first_try.h5")
# #print("name: ", pred.predict("online_val/fish_18/fish_000027320001_02895.png"))
# print("name: ", pred.predict("/Users/saul/programming/cnn/online_val/fish_17/fish_000023950001_02301.png"))

