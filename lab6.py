import csv
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network._base import ACTIVATIONS
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from sklearn.tree import DecisionTreeClassifier

datContent = []
with open('optdigits.dat', 'r') as f:
    d = f.readlines()
    for i in d[70:]:
        k = i.rstrip().split(',')
        datContent.append(k)

data = np.array(datContent, dtype=float)

X = data[:, :64]
y = data[:, 64]

results = []
clf = MLPClassifier(random_state=111)
kf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=111)
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train);

    y_pred = clf.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    results.append(score)

print('SAMO MLP DEFAULT')
print(results)
print(np.mean(results))
print(np.var(results))

results = []
layerSizes = [32, 64, 128, 256, 512]

# pętla iterująca po wielkościach wartstwy ukrytej
for layerSize in layerSizes:
    layerResults = []
    clf = MLPClassifier(random_state=111, hidden_layer_sizes=(layerSize))

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train);

        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        layerResults.append(score)

    results.append(layerResults)

print('ROZNA 1 WARSTWA')
print(results)
print(np.mean(results, axis=1))
print(np.var(results, axis=1))

clf = MLPClassifier(random_state=111,
                    hidden_layer_sizes=(512, 256, 256, 128, 64, 32, 16))
s_clf = KNeighborsClassifier()

s_results = []
results7 = []
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(X_train, y_train);
    s_clf.fit(X_train, y_train);

    y_pred = clf.predict(X_test)
    s_y_pred = s_clf.predict(X_test)

    score = accuracy_score(y_test, y_pred)
    s_score = accuracy_score(y_test, s_y_pred)

    results7.append(score)
    s_results.append(s_score)

print('SAMO KNN')
print(s_results)
print(np.mean(s_results))
print(np.var(s_results))

print('7 WARSTW')
print(results7)
print(np.mean(results7))
print(np.var(results7))

activator = ACTIVATIONS['relu']
for train_index, test_index in kf.split(X, y):
    y_train, y_test = y[train_index], y[test_index]
    output = np.copy(X)
    for l in range(clf.n_layers_ - 2):
        output = activator(
            np.matmul(output, clf.coefs_[l]) + clf.intercepts_[l])
        s_clf.fit(output[train_index], y_train)
        y_pred = s_clf.predict(output[test_index])
        base_score = accuracy_score(y_test, y_pred)
        print("  L%02i - %.3f - %i features"
              % (l, base_score, output.shape[1]))

pcaResults = []
pca = PCA(n_components=16)

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    pca.fit(X_train, y_train);
    extracted_X_train = pca.transform(X_train);
    s_clf.fit(extracted_X_train, y_train);

    extracted_X_test = pca.transform(X_test)
    y_pred = s_clf.predict(extracted_X_test)
    score = accuracy_score(y_test, y_pred)
    pcaResults.append(score)

print('PCA RESULTS')
print(pcaResults)
print(np.mean(pcaResults))
print(np.var(pcaResults))
