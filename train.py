import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
import codecs, json
from drawdata import draw_scatter
import pandas as pd
from sklearn.model_selection import train_test_split
from scikitplot.metrics import plot_roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
import seaborn as sns
import sklearn.metrics as metrics

def get_fp_tp(y, proba, threshold):
    """Возвращает количество долей ложно положительных и истинно положительных."""
    # Разносим по классам
    pred = pd.Series(np.where(proba>=threshold, 1, 0),
                    dtype='category')
    pred.cat.set_categories([0,1], inplace=True)
    # Создаём матрицу ошибок
    confusion_matrix = pred.groupby([y, pred]).size().unstack()\
                           .rename(columns={0: 'pred_0',
                                          1: 'pred_1'},
                                   index={0: 'actual_0',
                                          1: 'actual_1'})
    false_positives = confusion_matrix.loc['actual_0', 'pred_1']
    true_positives = confusion_matrix.loc['actual_1', 'pred_1']
    return false_positives, true_positives

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
obj_text = codecs.open('rgb.json', 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)
a_new = np.array(b_new)
print(a_new)
'''
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
# стандартизация входных данных
x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 2)
y_test_cat = keras.utils.to_categorical(y_test, 2)

#x_train = np.expand_dims(x_train, axis=3)
#x_test = np.expand_dims(x_test, axis=3)

print(x_train.shape)


model = keras.Sequential([
    Conv2D(64, (3,3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(128, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Conv2D(256, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(2,  activation='softmax')
])
#my_opt = tf.keras.optimizer.SGD(learning_rate=0.01, momentum=0.0, nesterov=True)
model.compile(optimizer='SGD',
             loss='binary_crossentropy',
             metrics=['accuracy'])

#x_train_split, x_val_split, y_train_split,
#y_val_split = train_test_split(x_train, y_train_cat, test_size=0.1)
his = model.fit(x_train, y_train_cat, batch_size=32,
 epochs=32, validation_split=0.1)

model.evaluate(x_test, y_test_cat)

# график качества
plt.plot(his.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Quality")
plt.grid(True)
plt.show()


# # вывод структуры НС в консоль
print(model.summary())

y_hat = model.predict(x_test)[:,1]
thresholds = np.linspace(0,1,100)
# defining fpr and tpr
tpr = []
fpr = []
# определяем положительные и отрицательные
'''
positives = np.sum(y_test==1)
negatives = np.sum(y_test==0)
# перебираем пороговые значения и получаем количество ложно и истинно положительных результатов 
for th in thresholds:
    fp,tp = get_fp_tp(y_test, y_hat, th)
    tpr.append(tp/positives)
    fpr.append(fp/negatives)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.plot(fpr,tpr, label="ROC Curve",color="blue")
plt.xlabel("False Positve Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()
'''
#Confusion Matrix






probs = model.predict(x_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(pd.to_numeric(y_test), preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


test_predictions = model.predict(x_test)
confusion = confusion_matrix(pd.to_numeric(y_test), np.argmax(test_predictions,axis=1))
confusion = pd.DataFrame(confusion, range(2),range(2))
plt.figure(figsize = (2,2))

sns.heatmap(confusion, annot=True, annot_kws={"size": 12}) # font size
plt.show()