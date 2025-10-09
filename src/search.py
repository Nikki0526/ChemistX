# -*- coding: utf-8 -*-
import pandas as pd
import google_colab_selenium as gs
import time
from pandas.core.frame import DataFrame # seems not used
import ast
from tensorflow.keras.layers import * # seems not used
from tensorflow.keras.models import * # seems not used
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import cv2
from imutils import paths
import tensorflow as tf
from keras.models import load_model
import joblib

"""
p_test String: path to SMILES test dataset
p_test_add String: path to additive test dataset
p_en String: path to pre-trained encoder model
p_dim_reducer String: path to pre-trained dimension reduecer model
p_model String: path to pre-trained model
"""
def search(p_test, p_test_add, p_en, p_dim_reducer, p_model):
  
    raw = pd.read_csv(p_test)
    names = list(raw.columns) # names = list(raw.columns)
  
    ## CIMG Interface
    
    #  Organize all SMILES into a list
    data = raw[names]
    smiles_all = []

    for i in range(len(names)):
      smiles_all.append(data[names[i]].drop_duplicates().tolist())

    smiles_all_flatten = sum(smiles_all, [])
    data_output = pd.DataFrame(smiles_all_flatten)
    data_output.columns = ['smiles']

    # Get CIMG vectors from interface
    wd = gs.ChromeDriver()
    # wd.quit()

    wd.get('http://cimg.dcaiku.com/')

    list_text = [] # Store CIMG Vectors
    list_textbox_value = [] # Store corresponding SMILES strings

    for i in range(len(data_output)):
      wd.find_element('id','smiles').clear() # Find and clear the input box
      element = wd.find_element('id','smiles')
      element.send_keys(data_output.at[i,'smiles']+'\n') # Input SMILES into the input box
      time.sleep(2)
      label = wd.find_elements('xpath',"//p") # Find the output

      list_text.append(label[0].text) # related output
      list_textbox_value.append(data_output.at[i,'smiles']) # origin smiles string

    # print(len(list_textbox_value))
    # print(len(list_text))

    ## Convert to DataFrame and clean

    from pandas.core.frame import DataFrame
    c = {"Text" : list_text,
       "Textbox_value" : list_textbox_value} # Convert lists into a dictionary
    cimg1 = DataFrame(c) # Convert dictionary to DataFrame
    cimg1['Text'] = cimg1['Text'].map(lambda x:str(x)[13:]) # Remove prefix "CIMG Vector: "
    cimg1 = cimg1[cimg1['Text']!=''] # Remove failed conversions (e.g., NaOH)
    cimg1 = cimg1.reset_index()
    del cimg1['index']

    # Convert string to list
    for i in range(len(cimg1)):
      #print(i)
      cimg1.at[i,'Text'] = ast.literal_eval(cimg1.at[i,'Text'])

    len(cimg1.at[0,'Text'])
    cimg2 = cimg1['Text'].apply(pd.Series)
    cimg2['smiles'] = cimg1['Textbox_value']

    # Fill missing CIMG vectors for additives (since Suzuki dataset lacks them, use the average)
    data_additive_mean = pd.read_csv(p_test_add)
    data_additive_mean.columns = cimg2.columns
    cimg3 = cimg2.append(data_additive_mean)
    cimg3 = cimg3.reset_index()
    del cimg3['index']

    # Merge CIMG vectors with SMILES data
    raw['additive_SMILES'] = 'none'
    names.append('additive_SMILES')
    data_cimg = pd.merge(raw,cimg3,left_on='ligand_SMILES',right_on='smiles',how='left')

    for i in range(len(names)-1):
      data_cimg = pd.merge(data_cimg,cimg3,left_on=names[i+1],right_on='smiles',how='left')
    data_cimg_final = data_cimg.loc[:,~data_cimg.columns.str.contains('smiles_')]

    ## Model prediction and output
    
    # Load pre-trained model
    tf.random.set_seed(666)
    np.random.seed(666)

    class UnitNormLayer(tf.keras.layers.Layer):
        '''Normalize vectors (euclidean norm) in batch to unit hypersphere.'''
        def __init__(self,**kwargs):
            super(UnitNormLayer, self).__init__()

        def call(self, input_tensor):
            norm = tf.norm(input_tensor, axis=1)
            return input_tensor / tf.reshape(norm, [-1, 1])

        def get_config(self):
            config = super(UnitNormLayer, self).get_config()
            return config

    # Embedding
    X_test = data_cimg_final.iloc[:,6:].values
    encoder_r = load_model(p_en, custom_objects={'UnitNormLayer': UnitNormLayer})
    encoded_vector_test=encoder_r.predict(X_test)
    if 'projector_z' in p_dim_reducer:
      projector_z = load_model(p_dim_reducer, custom_objects={'UnitNormLayer': UnitNormLayer})
      reduced_vector_test = projector_z.predict(encoded_vector_test)
    else: # PCA
      pca_model = joblib.load(p_dim_reducer)
      reduced_vector_test = pca_model.transform(encoded_vector_test)
    data_test = pd.DataFrame(reduced_vector_test)

    # Prediction
    model = joblib.load(p_model)
    if 'xgb' in model:
      predictions = model.predict(data_test)
    else: # RF models
      predictions = model.predict_proba(data_test)[:, 1]

    test = data_test.copy()
    test['y_pred'] = predictions
    # sorting
    test.sort_values(by='y_pred', inplace=True, ascending=False) # Sort by predicted values
    data_select = data_cimg_final.loc[test[:5].index]

    return data_select



