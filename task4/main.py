import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Defining constants
MODEL_DIR = './autoencoder'
OUTPUT_FILE = './output.csv'   
rerendermodel = True

# Load all the provided data
print('Loading data...')
pretrainFeatures = pd.read_csv('./pretrain_features.csv')
pretrainLabels = pd.read_csv('./pretrain_labels.csv')
trainFeatures = pd.read_csv('./train_features.csv')
trainLabels = pd.read_csv('./train_labels.csv')
testFeatures = pd.read_csv('./test_features.csv')
print('Data successfully loaded')

# Isolating the features and target data
pretrain_features = pretrainFeatures.iloc[:, 2:]
pretrain_target = pretrainLabels.iloc[:, 1]
# Taking the absolute value, so the network learns on positive labels
pretrain_target = pretrain_target.abs()
train_features = trainFeatures.iloc[:, 2:]
train_target = trainLabels.iloc[:, 1]
test_features = testFeatures.iloc[:,2:]
test_ID = testFeatures.iloc[:,0]

# Drop features with identical values for all datapoints (as they contain no useful information)
for i in pretrain_features.columns:
    if (pretrain_features[i][pretrain_features[i]==0].count() == 0 and  train_features[i][train_features[i]==0].count() == 0) or (pretrain_features[i][pretrain_features[i]==0].count() == 50000 and train_features[i][train_features[i]==0].count() == 50000):
        pretrain_features.drop([i], inplace=True, axis=1)
        train_features.drop([i], inplace=True, axis=1)
        test_features.drop([i], inplace=True, axis=1)

# Split into training and validation datasets
Xpre_train, Xpre_test, ypre_train, ypre_test = train_test_split(pretrain_features, pretrain_target, test_size=0.33, random_state=1)


# create model structure for autoencoder (encoder and decoder)
encoderModel = tf.keras.Sequential(name="encoder")
encoderModel.add(tf.keras.layers.InputLayer(pretrain_features.shape[1:]))
encoderModel.add(tf.keras.layers.Dense(760))
encoderModel.add(tf.keras.layers.Dropout(0.5))
encoderModel.add(tf.keras.layers.BatchNormalization())
encoderModel.add(tf.keras.layers.LeakyReLU())
encoderModel.add(tf.keras.layers.Dense(units=440))

decoderModel = tf.keras.Sequential(name="decoder")
decoderModel.add(tf.keras.layers.Dense(units=760))
decoderModel.add(tf.keras.layers.Dropout(0.5))
decoderModel.add(tf.keras.layers.BatchNormalization())
decoderModel.add(tf.keras.layers.LeakyReLU())
decoderModel.add(tf.keras.layers.Dense(pretrain_features.shape[1:][0]))


# load the autoencoder model if it always exists
if(os.path.exists(MODEL_DIR) and not rerendermodel):
    print('Loading Autoencoder...')
    autoencoder = tf.keras.models.load_model(MODEL_DIR)
    print('Autoencoder successfully loaded.')

# train the autoencoder model and save it
else:
    print('Training autoencoder model...')
    inp = Input(pretrain_features.shape[1:])
    code = encoderModel(inp)
    autoencoder_output = decoderModel(code)
    autoencoder = tf.keras.Sequential(name="autoencoder")
    autoencoder.add(encoderModel)
    autoencoder.add(decoderModel)
    
    # autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule), loss=losses.MeanSquaredError())
    autoencoder.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
    autoencoder.fit(x=Xpre_train, y=Xpre_train,validation_data=(Xpre_test,Xpre_test), batch_size=64, epochs=10, shuffle=True)
    autoencoder.save(MODEL_DIR)
    print('Autoencoder model created successfully')
    
print('Computing encoded features for pre-train, train- and test-features...')
encoderModel.summary()

# obtain freatures for pre-train, train and test sets from autoencoder
encoderOutputPreTrain = encoderModel.predict(pretrain_features)
encoderOutputTrain = encoderModel.predict(train_features)
encoderOutputTest = encoderModel.predict(test_features)

# split the new datasets into training and validation set
Xpre_trainb, Xpre_testb, ypre_trainb, ypre_testb = train_test_split(encoderOutputPreTrain, pretrain_target, test_size=0.33, random_state=1)
X_trainb, X_testb, y_trainb, y_testb = train_test_split(encoderOutputTrain, train_target, test_size=0.2, random_state=1)
print('Successfully created encoded features for pre-training, training and testing')


# create regression model, train it and predict from it
print('Creating regression model....')
regressionInputs = Input(encoderOutputPreTrain.shape[1:][0])
x = tf.keras.layers.Dropout(0.5)(regressionInputs)
x = tf.keras.layers.Dense(64)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(units=16)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(units=4)(x)
x = tf.keras.layers.Activation('relu')(x)
regressionOuput = tf.keras.layers.Dense(1)(x)

regressionModel = Model(inputs=regressionInputs, outputs=regressionOuput)
regressionModel.compile(optimizer="adam", loss="mean_squared_error", metrics=["mean_squared_error"])
regressionModel.fit(x=Xpre_trainb, y=ypre_trainb, validation_data=(Xpre_testb, ypre_testb) ,epochs=12, batch_size=64,  shuffle=True)

regressionModel.summary()
print('Regression model created successfully')


regressionModel.fit(x=X_trainb, y=y_trainb, validation_data=(X_testb,y_testb), epochs=700, batch_size=64, shuffle=True)

print('Predicting test labels...')
output = regressionModel.predict(encoderOutputTest)
outputDF = pd.DataFrame(output, columns = ['y'])
outputData = (test_ID.to_frame()).join(outputDF)
outputData.to_csv(OUTPUT_FILE, encoding='utf-8', index=False, float_format='%.5f')
print('Prediction completed successfully')
