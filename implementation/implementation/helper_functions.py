import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import TargetEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import PowerTransformer
import random
import statistics
import re
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# load_data()
def load_data(file_path):
    df = pd.read_csv(file_path)
    with open("df.pickle", "wb") as f:
      pickle.dump(df, f)
    return df

# split_data()
def split_data(df, tar_col, testsize, randomstate):
    X = df.drop(columns = tar_col)
    y = df[tar_col]
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size = testsize, random_state = randomstate)
    return xtrain, xtest, ytrain, ytest

def clean_data(df, imp_dict = None):
    # Start with the train_df
    # First, obtain the columns that have nulls and initiate a dictionary 
    if imp_dict == None:
        dict_col = {}

        # Get the list of all columns that have null values in the train dataset
        #df_nulls = df.columns[df.isna().any()].tolist() 

        # Get the list of all numerical columns that have nulls in train dataset
        df_nums = list((df.select_dtypes(exclude = object)).columns) 

        # Get the list of all object-type columns that have nulls in train dataset
        df_objs = list((df.select_dtypes(include = object)).columns)

        # Use SimpleImputer on numerical null columns via mean
        for col in df_nums:
            si_nums = SimpleImputer(strategy = "mean")
            si_nums.fit(df[[col]])
            dict_col[col] = si_nums
            df[[col]] = si_nums.transform(df[[col]])

        # Use SimpleImputer on object-type null columns via mode
        for col in df_objs:
            si_obj = SimpleImputer(strategy = "most_frequent")
            si_obj.fit(df[[col]])
            dict_col[col] = si_obj
            df[[col]] = si_obj.transform(df[[col]])

        # For pickling, create a dictionary that has the column names as keys and the simple imputer objects as values
        with open('imputation_dict.pickle', "wb") as f:
            pickle.dump(dict_col, f)
    else:
       for col in df:
          imp_type = imp_dict[col] # Extract the value from the dictionary we created
          df[[col]] = imp_type.transform(df[[col]])
          
          
    return df

def encode_data(df, tarcol, tartype, target_encoder = None):
    # First, extract the object-type columns
    df_obj_cols = list((df.select_dtypes(include = object)).columns)

    if target_encoder == None:

        te = TargetEncoder(target_type = tartype)
        te.fit(df[df_obj_cols], df[tarcol])
        df[df_obj_cols] = te.transform(df[df_obj_cols])

        with open("tar_enc.pickle", "wb") as f:
         pickle.dump(te, f)

    else:
       df[df_obj_cols] = target_encoder.transform(df[df_obj_cols])


    return df

def transform_model(df, tarcol = None, scaler = None):

  if scaler == None: 
    # Getting the dataset without the target
    df_no_tar = df.drop(columns = tarcol)
    df_no_tar_cols = list(df_no_tar.columns)

    # target type is not object type and it has 20 or more distinct value types
    if (tarcol != None) and (tarcol in df.columns) and (df[tarcol].dtype != 'object') and (df[tarcol].nunique() > 20):
        pt = PowerTransformer(method = 'yeo-johnson', standardize = False)
        pt.fit(df[[tarcol]])
        df[tarcol] = pt.transform(df[[tarcol]])

        with open("power_trans.pickle", "wb") as f:
            pickle.dump(pt, f)
        
    rs = RobustScaler()
    rs.fit(df_no_tar)
    df[df_no_tar_cols] = rs.transform(df_no_tar)

    with open("robust_scal.pickle", "wb") as f:
        pickle.dump(rs, f)
        
  else:
      df_cols = list(df.columns)
      df[df_cols] = scaler.transform(df[df_cols])
      print(df[df_cols])

  return df

# Make sure to train-test-split before training the model!
def train_model(xtrain, ytrain, xval, yval, alpha_level = None):
  if (ytrain.dtype != 'object') and (ytrain.nunique() > 20):

     l = Lasso(alpha = alpha_level)
     l.fit(xtrain, ytrain)

     preds = l.predict(xval)
     msqe = mean_squared_error(yval, preds)

     with open("lasso_model.pickle", "wb") as f:
        pickle.dump(l, f)
     return msqe
  else:
     xgb = XGBClassifier()
     xgb.fit(xtrain, ytrain)

     predics = xgb.predict(xval)

     test_proba = xgb.predict_proba(xval)
     auc_score = roc_auc_score(yval, test_proba[:,1])

     with open("xgbclass_model.pickle", "wb") as f:
        pickle.dump(xgb,f)
        
     print(classification_report(yval, predics))
     return auc_score