# diabetic retinopathy - DR
# https://archive.ics.uci.edu/ml/datasets/Diabetic+Retinopathy+Debrecen+Data+Set#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys
from sklearn.neighbors import KNeighborsClassifier
np.set_printoptions(threshold=sys.maxsize)

dataset = pd.read_csv('messidor_features.csv', header=None)

# rozdzielamy dane na: X - input data, Y - output data (do regresji)
X, Y = dataset.iloc[:, 0:19].values, dataset.iloc[:, 19].values

# przeskalowanie wektorów
x_std = StandardScaler().fit_transform(X)
print("\nx_std: ", x_std)

# macierz kowariancji dla input data
features = x_std.T
covariance_matrix = np.cov(features)
print("\ncovariance_matrix: ", covariance_matrix)

# obliczamy wartości własne i wektory własne z macierzy kowariancji
eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
print('\nEigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)
print("\n")

# obliczenia dla wyłonienia największych wartości własnych
tot = sum(eig_vals)
var_exp = [(i / tot) for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Dla 3 największych wartosci wlasnych uzyskujemy dokładnosc: ", cum_var_exp[2]*100, "%")

# plot explained variances
plt.bar(range(0, 19), var_exp, alpha=0.5,
        align='center', label='Wartość własna pojedynczego wektora')
plt.step(range(0, 19), cum_var_exp, where='mid',
         label='Zsumowane wartości własne')
plt.ylabel('Wartości własne')
plt.xlabel('Kolejne wektory własne')
plt.legend(loc='best')
plt.show()

# tworzymy zbiór wartości i przypisanych im wektorów w celu ich uporządkowania
eigen_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)
print("eigen_pairs: ", eigen_pairs)

# wybieram 3 wektory do utworzenia macierzy W
w = np.hstack((eigen_pairs[0][1][:, np.newaxis], eigen_pairs[1][1][:, np.newaxis], eigen_pairs[2][1][:, np.newaxis]))
print('Matrix W:\n', w)

# mnożymy cała początkową przeskalowaną macierz X przez macierz W
x_pca = x_std.dot(w)
print('Matrix x_pca:\n', x_pca)

# Regresja liniowa
model = LinearRegression().fit(x_pca, Y) # x_pca, Y

r_sq = model.score(x_pca, Y)
print('coefficient of determination:', r_sq)
print('b0:', model.intercept_)
print('b1,...,br:', model.coef_)

y_pred = model.predict(x_pca) # x_pca
print('predicted response:', y_pred, sep='\n')
y_pred_round = np.round(y_pred)
print('rounded predicted response:', y_pred_round)
print('oryginal response:', Y)

all_elem = len(Y)
match_elem = 0
for i in range(all_elem):
    if(y_pred_round[i] == Y[i]):
        match_elem += 1
print("percent match: ", 100*(match_elem/all_elem), "%" )

# k nearest neighbours
X_train, X_test, y_train, y_test = train_test_split(x_pca, Y, test_size=0.20)
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=20, metric='euclidean')
classifier.fit(X_train, y_train)

# y_pred = classifier.predict(x_pca[2].reshape(1, -1)) # [X_test[0]]
# for i in range(len(x_pca)):
#    print(classifier.predict(x_pca[2].reshape(1, -1)), " ")

y_pred = classifier.predict(X_test)
print("y_pred: ", y_pred)

all_elem = len(y_test)
match_elem = 0
for i in range(all_elem):
    if(y_pred[i] == y_test[i]):
        match_elem += 1

print("percent y element match: ", 100*(match_elem/all_elem), "%" )