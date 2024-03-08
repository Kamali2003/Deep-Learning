import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

dataset = pd.read_csv('/content/gender_classification_v7.csv')

dataset.isnull().sum()
dataset.gender.value_counts()
dataset.gender.replace({"Male":1, "Female":0}, inplace = True)
dataset.corr()

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

dataset_scaled = dataset.copy()
cols_to_scale = ["forehead_width_cm", "forehead_height_cm"]
dataset_scaled[cols_to_scale] = scaler.fit_transform(dataset_scaled[cols_to_scale])
dataset_scaled.head()

Y_s = dataset_scaled["gender"]
dataset_scaled.drop("gender",axis = "columns", inplace = True)
X_scaled = dataset_scaled.copy()
X_scaled.head()

X_train_s, X_test_s, Y_train_s, Y_test_s = train_test_split(X_scaled, Y_s, random_state = 0,  test_size = 0.25)

clf2 = MLPClassifier(solver='lbfgs', alpha=0.001, hidden_layer_sizes=(5,5,2), max_iter = 100,
activation = 'relu')

clf2.fit(X_train_s, Y_train_s)

Y_pred_s = clf2.predict(X_test_s)
Y_pred_train_s = clf2.predict(X_train_s)

print(Y_pred_s[:100])
print(Y_pred_train_s[:100])
print(accuracy_score(Y_test_s, Y_pred_s))
print(accuracy_score(Y_train_s, Y_pred_train_s))
print(precision_score(Y_test_s, Y_pred_s))
print(recall_score(Y_test_s, Y_pred_s))
print(f1_score(Y_test_s, Y_pred_s))

from sklearn import metrics
y_pred = clf2.predict(X_test_s)
fpr, tpr, _ = metrics.roc_curve(Y_test_s, Y_pred_s)
auc = metrics.roc_auc_score(Y_test_s, Y_pred_s)
print('area under the curve is ',auc)
plt.plot(fpr,tpr,label="3 neighbours , AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('auc_roc_curve')
plt.show()

from sklearn.model_selection import GridSearchCV
mlp_gs = MLPClassifier()
parameters = {
 'hidden_layer_sizes' : [(5,5,2), (15, 7,7,2), (10, 5, 2), (5, 2)],
 'activation': ['relu', 'logistic', 'tanh', 'identity'],
 'solver' : ['sgd', 'adam', 'lbfgs'],
 'learning_rate_init' : [0.01, 0.001, 0.05]
}

clf = GridSearchCV(mlp_gs, parameters, cv = 5)
clf.fit(X_train_s, Y_train_s)
print(clf.best_params_)
res = pd.DataFrame(clf.cv_results_)

res.head()
res = res.sort_values('rank_test_score')
res.to_csv('grid_CV_results.csv')
res[['param_activation', 'param_hidden_layer_sizes',
 'param_learning_rate_init', 'param_solver','mean_test_score',
 'std_test_score', 'rank_test_score' ]].head()
clf_best= MLPClassifier(solver='adam', hidden_layer_sizes=(10,5,2), activation = 'logistic',
learning_rate_init= 0.01)
clf_best.fit(X_train_s, Y_train_s)

Y_pred_s = clf_best.predict(X_test_s)
Y_pred_train_s = clf_best.predict(X_train_s)

print(accuracy_score(Y_test_s, Y_pred_s))
print(accuracy_score(Y_train_s, Y_pred_train_s))
print(precision_score(Y_test_s, Y_pred_s))
print(recall_score(Y_test_s, Y_pred_s))
print(f1_score(Y_test_s, Y_pred_s))

y_pred = clf_best.predict(X_test_s)

from sklearn import metrics
y_pred = clf2.predict(X_test_s)
fpr, tpr, _ = metrics.roc_curve(Y_test_s, Y_pred_s)
auc = metrics.roc_auc_score(Y_test_s, Y_pred_s)
print('area under the curve is ',auc)
plt.plot(fpr,tpr,label="3 neighbours , AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('auc_roc_curve')
plt.show()

