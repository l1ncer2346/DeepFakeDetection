import json
import numpy as np
import cv2
import os
import pandas as pd
import numpy as np

gf = {'dsg' : "2"}
y_train = np.array([], dtype = bool)

rgb_json = {}


with open("train.json", 'r') as train_json:
	train_data_json = json.load(train_json)

for filename in train_data_json.keys():
	y_train = np.append(y_train, train_data_json[filename][list(train_data_json[filename])[0]])

print(y_train)
current_path = os.getcwd()
counter = 0
x_train = np.zeros((len(train_data_json),32,32,3))

for filename in train_data_json.keys():
	#
	tmp_path = os.path.join(current_path, train_data_json[filename][list(train_data_json[filename])[1]])
	print(tmp_path)
	image = cv2.imread(train_data_json[filename][list(train_data_json[filename])[1]])
	height, width = x_train.shape[1:3]
	image = cv2.resize(image,x_train.shape[1:3],interpolation = cv2.INTER_AREA)
	for x in range(width):
		for y in range(height):
			rgb = []
			for color in range(3):
				rgb.append(image[x][y][color])
			#print(rgb)
			x_train[counter][x][y] = rgb
			#print(x_train[counter][x][y])

	counter+=1


filenames = list(train_data_json.keys())
height, width = x_train.shape[1:3]
for i in range(len(x_train)):
	rgb_json[str(filenames[i])] = {}
	print("Processing iteration... " + str(i))
	for row in range(width):
		pixel = row
		print("pixel number: " + str(pixel))
		for column in range(height):
			rgb_json[str(filenames[i])]["{0} pixel".format(pixel)] = {"red": str(x_train[i][row][column][0]), "green" : str(x_train[i][row][column][1]), "blue" : str(x_train[i][row][column][2])}

with open('x_train.json', 'w') as outfile:
    json.dump(rgb_json, outfile)


np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)