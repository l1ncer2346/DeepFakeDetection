import pandas as pd
import json

test_data_array = {}
train_data_array = {}
test_csv = pd.read_csv('test.csv')
test_data = test_csv.to_numpy()

train_csv = pd.read_csv('train.csv')
train_data = train_csv.to_numpy()

for i in range(len(train_data)):
	key = train_data[i][2]
	train_data_array[str(key)] = {"fake/real":str(train_data[i][3]), "path":"real_vs_fake\\real-vs-fake\\"+str(train_data[i][5]).replace('/','\\')}



for i in range(len(test_data)):
	key = test_data[i][2]
	test_data_array[str(key)] = {"fake/real":str(test_data[i][3]), "path":"real_vs_fake\\real-vs-fake\\"+str(train_data[i][5]).replace('/','\\')}


with open('test.json', 'w') as outfile:
    json.dump(test_data_array, outfile)

with open('train.json', 'w') as outfile:
    json.dump(train_data_array, outfile)
'''
y_train = []
for name in train_data:
	y_train.append(train_data[name])
'''
print(len(train_data_array))
print(len(test_data_array))
#print(train_csv.values)
#print(train_csv['id'][20000])