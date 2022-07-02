import pandas as pd
import sklearn.datasets as datasets
import sklearn.preprocessing as preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,BaggingClassifier
from sklearn.svm import SVC
import sklearn.cluster as cluster




#cancer_set = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',header=None)
#print(cancer_set.shape)
#cancer_feature = cancer_set.iloc[:, 2:]
#print(cancer_feature.shape)
#cancer_feature = cancer_feature.values
#print(type(cancer_feature))
#cancer_features_names = ['mean radius',
#'mean texture', 'mean perimeter','mean area', 'mean smoothness','mean compactness', 'mean concavity',
#'mean concave points', 'mean symmetry','mean fractal dimension','radius error','texture error','perimeter error',
#'area error', 'smoothness error','compactness error','concavity error','concave points error','symmetry error',
#'fractal dimension error','worst radius','worst texture', 'worst perimeter','worst area','worst smoothness',
#'worst compactness', 'worst concavity','worst concave points','worst symmetry','worst fractal dimension']
#cancer_target = cancer_set.iloc[:, 1]
#cancer_target = cancer_target.replace(['M', 'B'], [0, 1])
#cancer_target = cancer_target.values
#cancer_dataset = datasets.load_breast_cancer()
#print(cancer_dataset.data.shape)
#print(cancer_dataset.target.shape)
#standardizer = preprocessing.StandardScaler()
#standardizer = standardizer.fit(cancer_dataset.data)
#breast_cancer_standardization = standardizer.transform(cancer_dataset.data)
#print(breast_cancer_standardization.mean(axis=0))
#print(breast_cancer_standardization.std(axis=0))
#min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,10)).fit(cancer_dataset.data)
#breast_cancer_minmaxscaled = min_max_scaler.transform(cancer_dataset.data)
#max_abs_scaler = preprocessing.MaxAbsScaler().fit(cancer_dataset.data)
#breast_cancer_maxabsscaled = max_abs_scaler.transform(cancer_dataset.data)
#print(breast_cancer_maxabsscaled)
#normalizer = preprocessing.Normalizer(norm='l1').fit(cancer_dataset.data)
#breast_cancer_normalized = normalizer.transform(cancer_dataset.data)
#print(breast_cancer_normalized)
#binarizer = preprocessing.Binarizer(threshold=3.0).fit(cancer_dataset.data)
#breast_cancer_binarizer = binarizer.transform(cancer_dataset.data)
#print(breast_cancer_binarizer)
#labels = ['malignant', 'benign', 'malignant', 'benign']
#labelencoder =preprocessing.LabelEncoder()
#labelencoder = labelencoder.fit(labels)
#bc_labelencoded = labelencoder.transform(cancer_dataset.target_names)
#print(bc_labelencoded)
#imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean')
#imputer = imputer.fit(cancer_dataset.data)
#breast_cancer_imputed = imputer.transform(cancer_dataset.data)


def testscikit1():
    iris = datasets.load_iris()
    iris_norm = preprocessing.Normalizer(norm='l2').fit(iris.data)
    iris_normalized = iris_norm.transform(iris.data)
    #print(iris_normalized.mean(axis=0))
    #print(iris.target)
    iris_target_onehot = preprocessing.OneHotEncoder().fit_transform(iris.target.reshape(-1, 1))
    print(iris_target_onehot.toarray()[[0, 50, 100]])
    iris.data[:50, :] = np.nan
    iris_imputer = preprocessing.Imputer(missing_values='NaN', strategy='mean').fit(iris.data)
    iris_imputer = iris_imputer.transform(iris.data)
    print(iris_imputer.mean(axis=0))


#cancer = datasets.load_iris()
#nneighbor = []
#X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
#for i in range(3,11):
#    knn_classifier = KNeighborsClassifier(n_neighbors=i)
#    knn_classifier = knn_classifier.fit(X_train, Y_train)
#    print(knn_classifier.score(X_train, Y_train))
#    nneighbor.append(knn_classifier.score(X_train, Y_train))
#max_nneighbor = (max(nneighbor))
#n_neighbor = [index + 3 for index, value in enumerate(nneighbor) if value == max_nneighbor]
#print(n_neighbor)

def testscikit2():
    nneighbor = []
    iris = datasets.load_iris()
    boston = datasets.load_boston()
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, stratify=iris.target, random_state=30)
    print(X_train.shape)
    print(X_test.shape)
    knn_clf = KNeighborsClassifier()
    knn_clf = knn_clf.fit(X_train, Y_train)
    print(knn_clf.score(X_train, Y_train))
    print(knn_clf.score(X_test, Y_test))
    for i in range(3, 11):
        knn_classifier = KNeighborsClassifier(n_neighbors=i)
        knn_classifier = knn_classifier.fit(X_train, Y_train)
        y_pred = knn_classifier.predict(X_test)
        nneighbor.append(metrics.accuracy_score(Y_test, y_pred))
    max_nneighbor = (max(nneighbor))
    n_neighbor = [index + 3 for index, value in enumerate(nneighbor) if value == max_nneighbor]
    print(n_neighbor[0])


#cancer = datasets.load_breast_cancer()
#X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
#df_classifier = DecisionTreeClassifier(max_depth=2)
#df_classifier = df_classifier.fit(X_train, Y_train)
#print(df_classifier.score(X_train, Y_train))
#print(df_classifier.score(X_test, Y_test))


def testscikit3():
    dtrlist = []
    np.random.seed(100)
    boston = datasets.load_boston()
    X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, random_state=30)
    print(X_train.shape)
    print(X_test.shape)
    dt_reg = DecisionTreeRegressor()
    dt_reg = dt_reg.fit(X_train, Y_train)
    print(dt_reg.score(X_train, Y_train))
    print(dt_reg.score(X_test, Y_test))
    print(dt_reg.predict(X_test[:2]))
    for i in range(2, 6):
        dt_reg = DecisionTreeRegressor(max_depth=i).fit(X_train, Y_train)
        dtrlist.append(dt_reg.score(X_test, Y_test))
    dtrmax = max(dtrlist)
    dtrmaxlist = [index + 2 for index, value in enumerate(dtrlist) if value == dtrmax]
    print(dtrmaxlist[0])


def testscikit4():
    listreg = []
    np.random.seed(100)
    boston = datasets.load_boston()
    datasets.load_digits()
    X_train, X_test, Y_train, Y_test = train_test_split(boston.data, boston.target, random_state=30)
    print(X_train.shape)
    print(X_test.shape)
    rf_reg = RandomForestRegressor()
    rf_reg = rf_reg.fit(X_train, Y_train)
    print(rf_reg.score(X_train, Y_train))
    print(rf_reg.score(X_test, Y_test))
    print(rf_reg.predict(X_test[:2]))
    nestimator = 100
    for i in range(3, 6):
        dt_reg = RandomForestRegressor(max_depth=i, n_estimators=nestimator).fit(X_train, Y_train)
        listreg.append(dt_reg.score(X_test, Y_test))
    dtrmax = max(listreg)
    dtrmaxlist = [index + 2 for index, value in enumerate(listreg) if value == dtrmax]
    t = (dtrmaxlist[0], nestimator)
    print(t)


#cancer = datasets.load_breast_cancer()
#standardizer = preprocessing.StandardScaler()
#standardizer = standardizer.fit(cancer.data)
#cancer_standardized = standardizer.transform(cancer.data)
#X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, random_state=30)
#svm_classifier = SVC().fit(X_train, Y_train)
#print('Accuracy of Train Data :', svm_classifier.score(X_train, Y_train))
#print('Accuracy of Test Data :', svm_classifier.score(X_test, Y_test))


def testscikit5():
    digits = datasets.load_digits()
    X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, stratify=digits.target,
                                                        random_state=30)
    print(X_train.shape)
    print(X_test.shape)
    svm_clf = SVC().fit(X_train, Y_train)
    print(svm_clf.score(X_test, Y_test))
    digits_standardized = preprocessing.StandardScaler().fit(digits.data).transform(digits.data)
    X_train, X_test, Y_train, Y_test = train_test_split(digits_standardized, digits.target,
                                                        stratify=digits.target, random_state=30)
    svm_clf2 = SVC().fit(X_train, Y_train)
    print(svm_clf2.score(X_test, Y_test))


def testscikit6():
    iris = datasets.load_iris()
    X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target)
    km_cls = cluster.KMeans(n_clusters=3).fit(X_train)
    print(metrics.homogeneity_score(km_cls.predict(X_test), labels_pred=Y_test))
    agg_cls = cluster.AgglomerativeClustering(n_clusters=3)
    print(metrics.homogeneity_score(agg_cls.fit_predict(X_test), labels_pred=Y_test))
    af_cls = cluster.AffinityPropagation().fit(X_train)
    print(metrics.homogeneity_score(af_cls.predict(X_test), labels_pred=Y_test))
