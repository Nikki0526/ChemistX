# -*- coding: utf-8 -*-
import pandas as pd
import google_colab_selenium as gs
import time
from pandas.core.frame import DataFrame  
import ast
from tensorflow.keras.layers import *  
from tensorflow.keras.models import *  
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils import paths
import tensorflow as tf
from keras.models import load_model
import joblib
# ft_transformer
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import EarlyStopping
from tabtransformertf.models.fttransformer import FTTransformerEncoder, FTTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.decomposition import PCA
# xgboost
import xgboost as xgb
# Random Forests
from sklearn.ensemble import RandomForestRegressor

'''
p_train String: path to training dataset
'''
def load_data(p_train, p_train_add):
    data_projected_vector_0526 = pd.read_csv(p_train)
    data_projected_vector_0526[data_projected_vector_0526['type'] == 'nonsuzuki']  # training set
    data_wetlab = pd.read_csv(p_train_add)  
    data_wetlab['type'] = 'suzuki_wetlab'
    data_wetlab['yield'] = data_wetlab.index.tolist()
    data_wetlab['yield_number'] = data_wetlab.index.tolist()
    data_projected_vector = data_projected_vector_0526[data_projected_vector_0526['type'] == 'nonsuzuki'].append(
        data_wetlab)  
    data_projected_vector = data_projected_vector.reset_index()
    del data_projected_vector['index']
    return data_projected_vector

def model_rf_wetlab(p_train, p_train_add, fnum = 32):

    data_projected_vector = load_data(p_train, p_train_add)
    feature_num = 256
    feature = data_projected_vector.iloc[:, 0:feature_num]
    feature = feature.values

    X = feature

    # PCA for dimensionality reduction
    pca = PCA(n_components=fnum)
    principalComponents = pca.fit_transform(X)
    data_pca = pd.DataFrame(principalComponents)
    data_pca['label'] = data_projected_vector['yield_number'].tolist()

    # Data Splitting
    train_data, test_data = train_test_split(data_pca, test_size=100/4055, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(train_data.drop('label', axis=1), train_data['label'], test_size=0.2)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(test_data.drop('label', axis=1))
    y_test = test_data['label']

    # Random Forest Model
    model = RandomForestRegressor()

    # Model Training
    model.fit(X_train, y_train)

    # Model Prediction
    y_pred = model.predict(X_test)

    df = test_data
    df['y_pred'] = y_pred

    return model, pca, sc, df

def model_xgboost_wetlab(p_train, p_train_add, fnum = 32):

    data_projected_vector = load_data(p_train, p_train_add)

    data_all_copy = data_projected_vector.copy(deep=True)
    feature_num = 256
    feature = data_all_copy.iloc[:,0:feature_num]
    feature = feature.values

    X = feature

    # PCA for dimensionality reduction
    pca = PCA(n_components=fnum)
    principalComponents = pca.fit_transform(X)
    data_pca = pd.DataFrame(principalComponents)
    data_pca['label'] = data_all_copy['yield_number'].tolist()

    # Data Splitting
    train_data, test_data = train_test_split(data_pca, test_size=100/4055, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(train_data.drop('label', axis=1), train_data['label'], test_size=0.2)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(test_data.drop('label', axis=1))
    y_test = test_data['label']

    # XGBoost Model
    model = xgb.XGBRegressor(objective='reg:squarederror')

    # Model Training
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=16, verbose=True)

    # Model Prediction
    y_pred = model.predict(X_test)
    # res.append({'y_true': y_test, 'y_pred': y_pred})

    df = test_data
    df['y_pred'] = y_pred

    return model, pca, sc, df

def model_fttransformer_wetlab(p_train, p_train_add, fnum = 32):

    data_projected_vector = load_data(p_train, p_train_add)

    feature_num = 256
    feature = data_projected_vector.iloc[:,0:feature_num]
    feature = feature.values

    X = feature

    pca = PCA(n_components=fnum)
    principalComponents = pca.fit_transform(X)
    data_pca = pd.DataFrame(principalComponents)
    #data_pca = pd.DataFrame(feature) 
    data_pca['label'] = data_projected_vector['yield_number'].tolist()
    #train_data, test_data = train_test_split(data_pca, test_size=2112/6067, shuffle=False)
    train_data, test_data = train_test_split(data_pca, test_size=100/4055, shuffle=False)
    X_train, X_val = train_test_split(train_data, test_size=0.2)

    NUMERIC_FEATURES = list(range(fnum))

    sc = StandardScaler()
    X_train.loc[:, NUMERIC_FEATURES] = sc.fit_transform(X_train[NUMERIC_FEATURES])
    X_val.loc[:, NUMERIC_FEATURES] = sc.transform(X_val[NUMERIC_FEATURES])
    test_data.loc[:, NUMERIC_FEATURES] = sc.transform(test_data[NUMERIC_FEATURES])

    FEATURES = NUMERIC_FEATURES
    LABEL = 'label'

    # To TF Dataset
    train_dataset = df_to_dataset(X_train[FEATURES + [LABEL]], LABEL, shuffle=True)
    val_dataset = df_to_dataset(X_val[FEATURES + [LABEL]], LABEL, shuffle=False)  
    test_dataset = df_to_dataset(test_data[FEATURES + [LABEL]], LABEL, shuffle=False) 

    ft_linear_encoder = FTTransformerEncoder(
      numerical_features = FEATURES,
      categorical_features = [],
      numerical_data = X_train[FEATURES].values,
      categorical_data =None, # No categorical data
      y = None,
      #numerical_embedding_type='linear',
      numerical_embedding_type='periodic',
      numerical_bins=128,
      embedding_dim=64,
      depth=3,
      heads=6,
      attn_dropout=0.3,
      ff_dropout=0.3,
      explainable=True
    )

    # Pass th encoder to the model
    ft_linear_transformer = FTTransformer(
      encoder=ft_linear_encoder,
      out_dim=1,
      out_activation="relu",
    )

    LEARNING_RATE = 0.001
    # WEIGHT_DECAY = 0.0001
    NUM_EPOCHS = 500

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    ft_linear_transformer.compile(
      optimizer = optimizer,
      loss = {"output": tf.keras.losses.MeanSquaredError(name='mse'), "importances": None},
      metrics= {"output": [tf.keras.metrics.RootMeanSquaredError(name='rmse')], "importances": None},
    )

    early = EarlyStopping(monitor="val_output_loss", mode="min", patience=16, restore_best_weights=True)
    callback_list = [early]

    ft_linear_history = ft_linear_transformer.fit(
      train_dataset,
      epochs=NUM_EPOCHS,
      validation_data=val_dataset,
      callbacks=callback_list
    )

    y_pred = ft_linear_transformer.predict(test_dataset)
    df = test_data
    df['y_pred'] = y_pred['output'].ravel()

    return ft_linear_transformer, pca, sc, df

"""
p_train String: path to training dataset
p_train_add String: path to additive training dataset
p_test String: path to SMILES test dataset
p_test_add String: path to additive test dataset
p_en String: path to pre-trained encoder model
p_dim_reducer String: path to pre-trained dimension reduecer model
t_model String: type of required model
f_nums int: number of features used for training process
"""

def train_and_search(p_train, p_train_add, p_test, p_test_add, p_en, t_model, f_nums):
    raw = pd.read_csv(p_test)
    names = list(raw.columns)  # names = list(raw.columns)

    data = raw[names]
    smiles_all = []

    for i in range(len(names)):
        smiles_all.append(data[names[i]].drop_duplicates().tolist())

    smiles_all_flatten = sum(smiles_all, [])
    data_output = pd.DataFrame(smiles_all_flatten)
    data_output.columns = ['smiles']


    wd = gs.ChromeDriver()
    # wd.quit()

    wd.get('http://cimg.dcaiku.com/')

    list_text = []  
    list_textbox_value = []  

    for i in range(len(data_output)):
        wd.find_element('id', 'smiles').clear()  
        element = wd.find_element('id', 'smiles')
        element.send_keys(data_output.at[i, 'smiles'] + '\n')  
        time.sleep(2)
        label = wd.find_elements('xpath', "//p")  

        list_text.append(label[0].text)  # related output
        list_textbox_value.append(data_output.at[i, 'smiles'])  # origin smiles string

    # print(len(list_textbox_value))
    # print(len(list_text))


    from pandas.core.frame import DataFrame
    c = {"Text": list_text,
         "Textbox_value": list_textbox_value}  
    cimg1 = DataFrame(c)  
    cimg1['Text'] = cimg1['Text'].map(lambda x: str(x)[13:]) 
    cimg1 = cimg1[cimg1['Text'] != ''] 
    cimg1 = cimg1.reset_index()
    del cimg1['index']

    # str转list
    for i in range(len(cimg1)):
        # print(i)
        cimg1.at[i, 'Text'] = ast.literal_eval(cimg1.at[i, 'Text'])

    len(cimg1.at[0, 'Text'])
    cimg2 = cimg1['Text'].apply(pd.Series)
    cimg2['smiles'] = cimg1['Textbox_value']


    data_additive_mean = pd.read_csv(p_test_add)
    data_additive_mean.columns = cimg2.columns
    cimg3 = cimg2.append(data_additive_mean)
    cimg3 = cimg3.reset_index()
    del cimg3['index']


    raw['additive_SMILES'] = 'none'
    names.append('additive_SMILES')
    data_cimg = pd.merge(raw, cimg3, left_on='ligand_SMILES', right_on='smiles', how='left')

    for i in range(len(names) - 1):
        data_cimg = pd.merge(data_cimg, cimg3, left_on=names[i + 1], right_on='smiles', how='left')
    data_cimg_final = data_cimg.loc[:, ~data_cimg.columns.str.contains('smiles_')]


    tf.random.set_seed(666)
    np.random.seed(666)

    class UnitNormLayer(tf.keras.layers.Layer):
        '''Normalize vectors (euclidean norm) in batch to unit hypersphere.
        '''

        def __init__(self, **kwargs):
            super(UnitNormLayer, self).__init__()

        def call(self, input_tensor):
            norm = tf.norm(input_tensor, axis=1)
            return input_tensor / tf.reshape(norm, [-1, 1])

        def get_config(self):
            config = super(UnitNormLayer, self).get_config()
            return config

    # Embedding
    X_test = data_cimg_final.iloc[:, 6:].values
    encoder_r = load_model(p_en, custom_objects={'UnitNormLayer': UnitNormLayer})
    encoded_vector_test = encoder_r.predict(X_test)

    # Prediction
    if 'ft' in t_model:
        model, pca, sc, _ = model_fttransformer_wetlab(p_train, p_train_add, f_nums)
        projected_vector_test = pca.transform(encoded_vector_test)
        data_pca_test = pd.DataFrame(projected_vector_test)
        # Standardize Features
        data_pca_test.loc[:, :] = sc.transform(data_pca_test)
        data_test = df_to_dataset(data_pca_test, shuffle=False) # dataset_test
        predictions = model.predict(data_test)
        # Create a new DataFrame for the test data
        test = data_pca_test.copy()
        test['y_pred'] = predictions['output'].ravel()
    elif 'xgb' in t_model:
        model, pca, sc, _ = model_xgboost_wetlab(p_train, p_train_add, f_nums)
        projected_vector_test = pca.transform(encoded_vector_test)
        data_test = pd.DataFrame(projected_vector_test)
        scaled_data_test = sc.transform(data_test)
        predictions = model.predict(scaled_data_test)
        test = data_test.copy()
        test['y_pred'] = predictions
    else:  # Random Forests model
        model, pca, sc, _ = model_rf_wetlab(p_train, p_train_add, f_nums)
        projected_vector_test = pca.transform(encoded_vector_test)
        data_test = pd.DataFrame(projected_vector_test)
        scaled_data_test = sc.transform(data_test)
        predictions = model.predict(scaled_data_test)
        test = data_test.copy()
        test['y_pred'] = predictions

    # sorting
    test.sort_values(by='y_pred', inplace=True, ascending=False)  
    data_select = data_cimg_final.loc[test[:5].index]

    return data_select



