import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def plot_img(img, title):
    plt.title(title)
    plt.imshow(img, cmap='binary')
    plt.show()


def prep_img():
    img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    img = ~img
    img = cv2.blur(img, (10, 10))
    img = cv2.resize(img, (10, 10))
    img = np.array(img)
    img = img.reshape(-1)
    return img


main_path = "C:/Users/Kamil/Documents/projekty/rozpoznawanie_liter/litery"

df_trening = pd.read_csv(main_path + "/litery_trening/zbiór_treningowy.csv")
df_test = pd.read_csv(main_path + "/litery_test/zbiór_testowy.csv")

test = pd.read_csv(main_path + "/data_test.csv")
test = np.array(test)

X_train = []
Y_train = []
X_test = []
Y_test = []

df_trening = df_trening.reset_index()
for index, row in df_trening.iterrows():
    img_name = main_path + "/litery_trening/" + row["nazwa"]
    img_klasa = row['klasa']
    img = prep_img()
    X_train.append(img)
    Y_train.append(img_klasa)

df_test = df_test.reset_index()
for index, row in df_test.iterrows():
    img_name = main_path + "/litery_test/" + row["nazwa"]
    img_klasa = row['klasa']
    img = prep_img()
    X_test.append(img)
    Y_test.append(img_klasa)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

classifier = MLPClassifier(hidden_layer_sizes=(500), activation='logistic', alpha=1e-4,
                           solver='sgd', tol=1e-4, random_state=1, learning_rate_init=.1, verbose=True)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print("Dokładność: {0}".format(round(accuracy_score(Y_test, Y_pred)*100, 3)) + "%")

Y_pred = classifier.predict(test)
print(Y_pred)
