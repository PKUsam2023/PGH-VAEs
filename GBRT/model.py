from sklearn.ensemble import GradientBoostingRegressor as GBR
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.externals import joblib
import random

def GBRT_model(x_train, y_train, x_valid, y_valid, learning_rate, n_estimators, subsample, min_samples_split, max_depth):
    X_train = np.array(x_train)
    y_train = np.array(y_train)
    X_test = np.array(x_valid)
    y_test = np.array(y_valid)

    GBRTmodel = train_GBR_model(X_train, y_train, learning_rate, n_estimators, subsample, min_samples_split, max_depth)
    MAEx, MSEx, RMSEx, MAE, MSE, RMSE = test_model(X_train, y_train, X_test, y_test, GBRTmodel)

    return MAEx, MSEx, RMSEx, MAE, MSE, RMSE

def train_GBR_model(feature_set, label_set, learning_rate, n_estimators, subsample, min_samples_split, max_depth):

    model = GBR(loss='ls', learning_rate=learning_rate, n_estimators=n_estimators,
                subsample=subsample, min_samples_split=min_samples_split, max_depth=max_depth)
    model.fit(feature_set, label_set)
    joblib.dump(model, './model1.pkl')
    return model


def test_model(X_train, y_train, X_test, y_test, model):
    X_train = np.array(X_train, dtype='float32')
    y_train = np.array(y_train, dtype='float32')
    pred_train = model.predict(X_train)
    MAEx = metrics.mean_absolute_error(y_train, pred_train)
    MSEx = metrics.mean_squared_error(y_train, pred_train)
    RMSEx = MSEx ** 0.5

    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='float32')
    pred_test = model.predict(X_test)
    MAE = metrics.mean_absolute_error(y_test, pred_test)
    MSE = metrics.mean_squared_error(y_test, pred_test)
    RMSE = MSE ** 0.5

    return MAEx, MSEx, RMSEx, MAE, MSE, RMSE
