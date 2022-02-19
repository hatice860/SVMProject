import os
import re
import string
import os
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt




dataset = pd.read_csv('SosyalMedyaReklamKampanyasi.csv')


nok_isaretleri_kümesi = string.punctuation
# etkisiz kelimeler
etkisiz_kelimeler_kümesi = stopwords


def clean_data(data):
    data = data.lower()
    data = data.replace("\\n", " ")
    data = re.sub("[0-9]+", " ", data)
    data = re.sub(r'[^\w\s]', '', data)
    data = re.sub("'(\w+)", "", data)
    data = " ".join(list(map(lambda x: x if x not in nok_isaretleri_kümesi else " ", data)))
    data = " ".join([i for i in data.split() if i not in etkisiz_kelimeler_kümesi])
    data = " ".join([i for i in data.split() if len(i) > 1])
    return data


X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
print(dataset)

#veriyi egitim ve test olarak ayırdık
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#bagımsız degiskenlerden yas ile tahmini gelir aynı birimde olmadıgı için
#feature scaling uygulayacağız
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#svm modeli olusturmak ve egitmek
from sklearn.svm import SVC
classifier=SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)

#test seti ile tahmin yapmak
y_pred = classifier.predict(X_test)

#hata matrisini oluşturma
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)

#grafik
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
     plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                 c = ListedColormap(('yellow', 'green'))(i), label = j)
plt.title('SVM (Eğitim Seti)')
plt.xlabel('Yaş')
plt.ylabel('Maaş')
plt.legend()
plt.show()
