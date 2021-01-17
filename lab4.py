import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import DistanceMetric
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
np.set_printoptions(suppress=True)

# Pobieramy zestaw danych
digits = load_digits()
# Obrazy
images = digits.images
# Etykiety
y = digits.target

# Spłaszczenie obrazów do wektora
X = images.reshape((images.shape[0], -1))
print(X.shape)

#Podział na zbiór treningowy i testowy stratyfikowany na 5 foldów
kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1410)
results = []
# iterowanie po zbiorach treningowych i testowych po podzieleniu na foldy
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #ekstrakcja cech PCA
    pca = PCA(n_components=X_train.shape[1], random_state=1410)
    pca.fit(X_train)

    # procent objasnioniej wariacji
    evr = pca.explained_variance_ratio_
    evr_acc = np.add.accumulate(evr)
    print(evr_acc)

    localResults = []
    #iterowanie po kolejnych częściach wariancji jakie mają być objaśnione
    for i in [0.2,0.3,0.4,0.5,0.6]:
        #indeks wartosci najblizszej dla kazdego z procenta objaśnianej wariancji
        index = np.abs(evr_acc - i).argmin()

        X_trainPCA = pca.transform(X_train)[:, :index + 1 ]
        X_testPCA = pca.transform(X_test)[:, :index + 1]

        # Klasyfikacja 1-NN
        # Obliczanie dystansu
        dm = DistanceMetric.get_metric("euclidean")
        # Dystanse test X train
        # print(X_train.shape, X_test.shape)
        distances = dm.pairwise(X_testPCA, X_trainPCA)
        # Jeden najblizszy sasiad
        neighbors = np.squeeze(np.argsort(distances, axis=1)[:, :1])
        y_pred = y_train[neighbors]
        # Ocena
        score = accuracy_score(y_test, y_pred)
        # Zapisanie wyniku dla danego procenta
        localResults.append(score)

    #wyniki dla danego folda
    results.append(localResults)

results = np.array(results)
print(results)
# średnia dla każdego z wybranych procentów objaśnianej wariancji
means = np.mean(results, axis=0)
print('Eksperyment 1', means)


#Eksperyment 2

results = []
# ekstrakcja cech PCA na całym zbiorze
pca = PCA(n_components=X.shape[1], random_state=1410)
pca.fit(X)

# procent objasnioniej wariacji
evr = pca.explained_variance_ratio_
evr_acc = np.add.accumulate(evr)

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    localResults = []
    #iterowanie po kolejnych częściach wariancji jakie mają być objaśnione
    for i in [0.2,0.3,0.4,0.5,0.6]:
        #indeks wartosci najblizszej dla kazdego z procentu wariancji
        index = np.abs(evr_acc - i).argmin()

        X_trainPCA = pca.transform(X_train)[:, :index + 1 ]
        X_testPCA = pca.transform(X_test)[:, :index + 1]

        # Klasyfikacja 1-NN
        # Obliczanie dystansu
        dm = DistanceMetric.get_metric("euclidean")
        # Dystanse test X train
        distances = dm.pairwise(X_testPCA, X_trainPCA)
        # Jeden najblizszy sasiad
        neighbors = np.squeeze(np.argsort(distances, axis=1)[:, :1])
        y_pred = y_train[neighbors]
        # Ocena
        score = accuracy_score(y_test, y_pred)
        localResults.append(score)

    results.append(localResults)

results = np.array(results)
# średnia dla każdego z wybranych procentów objaśnianej wariancji
means = np.mean(results, axis=0)
print('Eksperyment 2', means)


#Eksperyment 3
pca = PCA(n_components=X.shape[1], random_state=1410)
pca.fit(X)

X_PCA = pca.transform(X)[:, :2]

# Klasyfikacja 1-NN
# Obliczanie dystansu
dm = DistanceMetric.get_metric("euclidean")
# Dystanse między wszystkimi próbkami (X x X)
distances = dm.pairwise(X_PCA, X_PCA)
# Jeden najblizszy sasiad
neighbors = np.squeeze(np.argsort(distances, axis=1)[:, :1])
y_pred = y[neighbors]
# Ocena
score = accuracy_score(y, y_pred)
print('Eksperyment 3', score)


#Eksperyment 4

results = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Klasyfikacja 1-NN
    # Obliczanie dystansu
    dm = DistanceMetric.get_metric("euclidean")
    # Dystanse test X train
    distances = dm.pairwise(X_test, X_train)
    # Jeden najblizszy sasiad
    neighbors = np.squeeze(np.argsort(distances, axis=1)[:, :1])
    y_pred = y_train[neighbors]
    # Ocena
    score = accuracy_score(y_test, y_pred)
    results.append(score)

results = np.array(results)
# średnia z walidacji krzyżowej
mean = np.mean(results)
print('Eksperyment 4', mean)