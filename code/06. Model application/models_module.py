import sklearn
import numpy as np
import matplotlib.pyplot as plt


def cont_train_test_X_y_split(data, features, test_size=0.2, random_state=None):
    df_model = data[data['sentiment_score'] != 0].reset_index()
    df_model = df_model[['log_ret'] + features]
    X = df_model[features]
    y = df_model[['log_ret']].values.ravel()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size,
                                                                                random_state=random_state)
    return X_train, X_test, y_train, y_test


def discr_train_test_X_y_split(data, features, test_size=0.2, random_state=None):
    df_model = data[data['sentiment_score'] != 0].reset_index()
    df_model['log_ret'] = [0 if value < 0 else 1 for value in df_model['log_ret']]
    df_model = df_model[['log_ret'] + features]
    X = df_model[features]
    y = df_model[['log_ret']].values.ravel()
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=test_size,
                                                                                random_state=random_state)
    return X_train, X_test, y_train, y_test


def min_max_scaler(X_train, X_test):
    scaler = sklearn.preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def LogisticReg(X_train, y_train, X_test, y_test, random_state=None):
    model = sklearn.linear_model.LogisticRegression(random_state=random_state)
    grid_values = {'penalty': ['l1', 'l2'],
                   'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                   'solver': ['liblinear'],
                   'max_iter': [100000]}
    clf = sklearn.model_selection.GridSearchCV(model, param_grid=grid_values, verbose=0, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    auc = round(sklearn.metrics.auc(fpr, tpr), 3)
    # Plot of the ROC Curve
    plt.rcParams["figure.figsize"] = (12,7)
    plt.plot(fpr, tpr, '#8635D5', label='Log. Reg. AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return auc

def RandomForestClas(X_train, y_train, X_test, y_test, random_state=None):
    model = sklearn.ensemble.RandomForestClassifier(random_state=random_state)
    grid_values = {'n_estimators': [200],
                   'max_features': ['sqrt', 'log2'],
                   'max_depth': [4, 5],
                   'criterion': ['gini', 'entropy']}
    clf = sklearn.model_selection.GridSearchCV(model, param_grid=grid_values, verbose=0, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    auc = round(sklearn.metrics.auc(fpr, tpr), 3)
    # Plot of the ROC Curve
    plt.rcParams["figure.figsize"] = (12,7)
    plt.plot(fpr, tpr, '#AD3960', label='Rand. Forest AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return auc


def DecisionTreeClas(X_train, y_train, X_test, y_test, random_state=None):
    model = sklearn.tree.DecisionTreeClassifier(random_state=random_state)
    grid_values = {'max_leaf_nodes': list(range(2, 100)),
                   'min_samples_split': [2, 3, 4]}
    clf = sklearn.model_selection.GridSearchCV(model, param_grid=grid_values, verbose=0, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    auc = round(sklearn.metrics.auc(fpr, tpr), 3)
    # Plot of the ROC Curve
    plt.rcParams["figure.figsize"] = (12,7)
    plt.plot(fpr, tpr, '#F24982', label='Dec. Tree AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return auc


def NaiveBayes(X_train, y_train, X_test, y_test):
    model = sklearn.naive_bayes.GaussianNB()
    grid_values = {'var_smoothing': np.logspace(0, -9, num=100)}
    clf = sklearn.model_selection.GridSearchCV(model, param_grid=grid_values, verbose=0, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    auc = round(sklearn.metrics.auc(fpr, tpr), 3)
    # Plot of the ROC Curve
    plt.rcParams["figure.figsize"] = (12,7)
    plt.plot(fpr, tpr, '#F9C823', label='N. Bayes AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return auc


def SVC(X_train, y_train, X_test, y_test, random_state=None):
    model = sklearn.svm.SVC(random_state=random_state, probability=True)
    grid_values = {#'C': [1, 10, 100, 1000],
                   'gamma': [1, 0.1, 0.001],
                   'kernel': ['linear', 'rbf']}
    clf = sklearn.model_selection.GridSearchCV(model, param_grid=grid_values, verbose=0, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    auc = round(sklearn.metrics.auc(fpr, tpr), 3)
    # Plot of the ROC Curve
    plt.rcParams["figure.figsize"] = (12,7)
    plt.plot(fpr, tpr, '#2DC574', label='SVC AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return auc


def KNN(X_train, y_train, X_test, y_test):
    model = sklearn.neighbors.KNeighborsClassifier()
    grid_values = {'n_neighbors': list(range(1, 31))}
    clf = sklearn.model_selection.GridSearchCV(model, param_grid=grid_values, verbose=0, cv=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
    auc = round(sklearn.metrics.auc(fpr, tpr), 3)
    # Plot of the ROC Curve
    plt.rcParams["figure.figsize"] = (12,7)
    plt.plot(fpr, tpr, '#006CDC', label='KNN AUC = %0.2f' % auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    return auc
