from catboost import CatBoostClassifier
from collections import Counter
import math
from math import atan2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import pickle
from PIL import Image
from pylab import rcParams
# import scipy
import scipy.stats as stats
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs, load_wine, load_breast_cancer, load_digits, load_iris, load_diabetes
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import chi2, RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, HuberRegressor, Lasso, ElasticNet
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error, confusion_matrix
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold, GridSearchCV, RepeatedStratifiedKFold, train_test_split, RepeatedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import statistics
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import glm
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportions_chisquare, proportions_ztest
import sys
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Set the logging level to ERROR
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.metrics import RootMeanSquaredError, MeanSquaredError, MeanAbsoluteError
from ucimlrepo import fetch_ucirepo
from xgboost import XGBClassifier, XGBRegressor

rcParams['figure.figsize'] = 10, 5
pd.set_option('display.max_rows', 95000)
pd.set_option('display.max_columns', 5000)
pd.set_option('display.width', 10000)
pd.set_option('display.max_colwidth', 10000)
np.set_printoptions(threshold=np.inf)

