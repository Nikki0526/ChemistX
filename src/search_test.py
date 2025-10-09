# -*- coding: utf-8 -*-
import pandas as pd
import google_colab_selenium as gs
import time
from pandas.core.frame import DataFrame  # seems not used
import ast
from tensorflow.keras.layers import *  # seems not used
from tensorflow.keras.models import *  # seems not used
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils import paths
import tensorflow as tf
from keras.models import load_model
import joblib

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

"""
p_raw String: path to raw SMILES dataset
p_add String: path to additive dataset
p_en String: path to pre-trained encoder model
p_dim_reducer String: path to pre-trained dimension reduecer model
p_model String: path to pre-trained model
"""

def search(p_raw, p_add, p_en, p_dim_reducer, p_model):
    raw = pd.read_csv(p_raw)
    names = list(raw.columns)  # names = list(raw.columns)

    ## CIMG Interface

    # Collect all SMILES into one list
    data = raw[names]
    smiles_all = []

    for i in range(len(names)):
        smiles_all.append(data[names[i]].drop_duplicates().tolist())

    smiles_all_flatten = sum(smiles_all, [])
    data_output = pd.DataFrame(smiles_all_flatten)
    data_output.columns = ['smiles']

    # Retrieve CIMG vectors from the interface
    wd = gs.ChromeDriver()
    # wd.quit()

    wd.get('http://cimg.dcaiku.com/')

    list_text = []  # Store the CIMG vectors
    list_textbox_value = []  # Store the corresponding SMILES strings

    for i in range(len(data_output)):
        wd.find_element('id', 'smiles').clear()  # Locate the input box and clear it
        element = wd.find_element('id', 'smiles')
        element.send_keys(data_output.at[i, 'smiles'] + '\n')  # Input the string into the input box
        time.sleep(2)
        label = wd.find_elements('xpath', "//p")  # Locate the output

        list_text.append(label[0].text)  # related output
        list_textbox_value.append(data_output.at[i, 'smiles'])  # origin smiles string

    # print(len(list_textbox_value))
    # print(len(list_text))

    ## Concert to dataframe and clean up

    from pandas.core.frame import DataFrame
    c = {"Text": list_text,
         "Textbox_value": list_textbox_value}  # Convert lists into a dictionary
    cimg1 = DataFrame(c)  # Convert the dictionary into a DataFrame
    cimg1['Text'] = cimg1['Text'].map(lambda x: str(x)[13:])  # Remove the "CIMG Vector: " prefix
    cimg1 = cimg1[cimg1['Text'] != '']  # Remove failed conversions such as NaOH
    cimg1 = cimg1.reset_index()
    del cimg1['index']

    # Convert string to list
    for i in range(len(cimg1)):
        # print(i)
        cimg1.at[i, 'Text'] = ast.literal_eval(cimg1.at[i, 'Text'])

    len(cimg1.at[0, 'Text'])
    cimg2 = cimg1['Text'].apply(pd.Series)
    cimg2['smiles'] = cimg1['Textbox_value']

    # Fill in missing CIMG vectors for additives (since the Suzuki dataset lacks additives, use the mean vector)
    data_additive_mean = pd.read_csv(p_add)
    data_additive_mean.columns = cimg2.columns
    cimg3 = cimg2.append(data_additive_mean)
    cimg3 = cimg3.reset_index()
    del cimg3['index']

    # Merge the CIMG vectors with the SMILES data
    raw['additive_SMILES'] = 'none'
    names.append('additive_SMILES')
    data_cimg = pd.merge(raw, cimg3, left_on='ligand_SMILES', right_on='smiles', how='left')

    for i in range(len(names) - 1):
        data_cimg = pd.merge(data_cimg, cimg3, left_on=names[i + 1], right_on='smiles', how='left')
    data_cimg_final = data_cimg.loc[:, ~data_cimg.columns.str.contains('smiles_')]

    ## Perform model prediction and output results

    # Load the pre-trained model
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
    if 'projector_z' in p_dim_reducer:
        projector_z = load_model(p_dim_reducer, custom_objects={'UnitNormLayer': UnitNormLayer})
        projected_vector_test = projector_z.predict(encoded_vector_test)
    else:  # PCA
        pca_model = joblib.load(p_dim_reducer)
        projected_vector_test = pca_model.transform(encoded_vector_test)
    data_test = pd.DataFrame(projected_vector_test)

    # Prediction
    if 'ft' in p_model:
        fnum = 32
        FEATURES = list(range(fnum))
        # Recreating the encoder as in the original model
        ft_linear_encoder = FTTransformerEncoder(
            numerical_features=FEATURES,
            categorical_features=[],
            numerical_data=None,
            categorical_data=None,  # No categorical data
            y=None,
            # numerical_embedding_type='linear',
            numerical_embedding_type='periodic',
            numerical_bins=128,
            embedding_dim=64,
            depth=3,
            heads=6,
            attn_dropout=0.3,
            ff_dropout=0.3,
            explainable=True
        )

        # Recreating the FTTransformer model
        ft_linear_transformer = FTTransformer(
            encoder=ft_linear_encoder,
            out_dim=1,
            out_activation="relu",
        )

        # Compile the model (if necessary for your use case)
        LEARNING_RATE = 0.001
        optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        ft_linear_transformer.compile(
            optimizer=optimizer,
            loss={"output": tf.keras.losses.MeanSquaredError(name='mse'), "importances": None},
            metrics={"output": [tf.keras.metrics.RootMeanSquaredError(name='rmse')], "importances": None},
        )

        data_pca_test = pd.DataFrame(projected_vector_test)
        # Standardize Features
        sc = joblib.load('fttransformer/sc_ft.joblib')
        data_pca_test.loc[:, :] = sc.transform(data_pca_test)
        data_test = df_to_dataset(data_pca_test, shuffle=False)  # dataset_test

        # Initialize dummy data
        dummy_data = np.zeros((1, fnum))  # 1 row, fnum features
        dummy_data = sc.transform(dummy_data)  # Apply the same StandardScaler used for real data

        # Convert to TensorFlow Dataset
        dummy_dataset = tf.data.Dataset.from_tensor_slices(dummy_data)
        dummy_dataset = dummy_dataset.batch(1)

        ft_linear_transformer(dummy_dataset)
        ft_linear_transformer.load_weights(p_model)
        predictions = ft_linear_transformer.predict(data_test)
        # Create a new DataFrame for the test data
        test = data_pca_test.copy()
        test['y_pred'] = predictions['output'].ravel()
    elif 'xgb' in model:
        model = joblib.load(p_model)
        predictions = model.predict(data_test)
        test = data_test.copy()
        test['y_pred'] = predictions
    else:  # RF models
        model = joblib.load(p_model)
        predictions = model.predict_proba(data_test)[:, 1]
        test = data_test.copy()
        test['y_pred'] = predictions

    # sorting
    test.sort_values(by='y_pred', inplace=True, ascending=False)  # Sort by predicted values
    data_select = data_cimg_final.loc[test[:5].index]

    return data_select



