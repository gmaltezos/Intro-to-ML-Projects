import numpy as np
import pandas as pd
import tensorflow as tf
import os, glob
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input


# Global variables
imagePaths = []
featureSource = './featuresEfficientB7.csv' # where will the features be saved / can be fetched from
outputFile = './outputB7.txt'               # save the output to this file
newSize = (300, 300)                        # dimension all images will be resized to


# 1) Resizing the images to uniform dimension
print("Resizing Images...")
if os.path.exists('./resized_images/'):
    # If the images have already been resized previously, skip the resizing step
    print("Images have already been resized, skipping this step.")
    imagePaths = np.array(glob.glob('./resized_images/' + '/*.jpg'))
    
else:
    os.makedirs('./resized_images/')
    imagePathArray = np.array(glob.glob('./food' + '/*.jpg'))

    for imagePath in tqdm(imagePathArray):
        
        # Resize the image using the tensorflow preprocessing pipeline
        image = tf.keras.preprocessing.image.load_img(imagePath)
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.resize_with_pad(image=image, target_height=newSize[0], target_width=newSize[1], antialias=True)
        image = tf.keras.preprocessing.image.array_to_img(image)

        # Save the resized images so that they do not have to be recomputed during every re-run of the algorithm
        newImgPath = imagePath.replace("food", "resized_images")
        image.save(newImgPath)
    
    print("Resizing of images successfully completed")

# save all image names in an array - caution: the paths are not ordered
imagePaths = np.array(glob.glob('./resized_images/' + '/*.jpg'))



# 2) Extract features / dense representations for every image with some pre-trained cnn and store them in a separate file
featuremodel = tf.keras.applications.efficientnet.EfficientNetB7(weights="imagenet", include_top=False, pooling='avg', input_shape=(newSize[0], newSize[1], 3))
featuremodel.trainable = False
# feature_saved = np.empty((10000,512)) # VGG16
feature_saved = np.empty((10000,2560)) # EfficientNetB7

print("Extracting features for all images...")
if(os.path.exists(featureSource)):
    # Load features if they have already been computed earlier
    print("Saved features found, loading...")
    feature_saved = np.loadtxt(featureSource)
    print("Features loaded successfully")
else:
    i=0
    for imagePath in tqdm(imagePaths):
        image = tf.keras.preprocessing.image.load_img(imagePath)
        image = tf.keras.preprocessing.image.img_to_array(image)

        # Normalize the image data data
        image = np.expand_dims(image, axis=0)
        # image = tf.keras.applications.vgg16.preprocess_input(image) # VGG16
        image = tf.keras.applications.efficientnet.preprocess_input(image) # EfficientNetB7
        features = featuremodel.predict(image)

        # save the features ordered by image number
        idx = imagePath[17:22]
        idx = int(idx)
        feature_saved[idx,:] = features
        i=i+1
    print("Feature extraction completed successfully")

    # Save the features so that they do not have to be recomputed on re-run
    print("Saving Features...")
    np.savetxt(featureSource, feature_saved)
    print("Features saved successfully")


# 3) Preprocess input and test triplets into tensors that can be fed to a neural net
# Create matrix with combinations and expected output and the inverted order with opposite labels
print("Loading training and test triplets...")
# load the train and test triplets
orderedTrainTriplets = pd.read_csv("./train_triplets.txt", delim_whitespace=True, header=None, names=["reference", "positive", "negative"])
testTriplets = pd.read_csv("./test_triplets.txt", delim_whitespace=True, header=None, names=["reference", "positive", "negative"])

# Use all triplets also in reverse order to train for the negative case. Otherwise the network would learn to return always 1
invertedTrainTriplets = orderedTrainTriplets.reindex(columns=["reference", "negative", "positive"])
invertedTrainTriplets.rename(columns = {'reference':'reference', 'negative':'positive', 'positive': 'negative'}, inplace = True)
orderedTrainTriplets["label"] = "1"
invertedTrainTriplets["label"] = "0"

# Build the train triplet tensor from the train file as well as the inverted ones through concatenation
trainTriplets = pd.concat([orderedTrainTriplets, invertedTrainTriplets], ignore_index=True)

# Randomize the order of the triplets (caution: this tensor contains the image names as triplets, not the features)
trainTriplets = trainTriplets.sample(frac=1)
print("Triplets successfully loaded (and reordered where required)")

# Build training tensors with the extracted features instead of image names for the model to be applied to:
print("Building training tensor with labels...")
trainingTensor = []
labels = []
numTrainTriplets = len(trainTriplets)

for k in tqdm(range(numTrainTriplets)):
    # Select the kth triplet in the training triplet tensor
    triplet = trainTriplets.iloc[k]

    # Get the features of all three images in the triplet and concatenate them
    reference = feature_saved[triplet["reference"]]
    positive = feature_saved[triplet["positive"]]
    negative = feature_saved[triplet["negative"]]
    triplet_tensor = np.concatenate((reference, positive, negative), axis=-1)

    # Add the resulting triplet of features and its label to the training tensor 
    trainingTensor.append(triplet_tensor)
    labels.append(int(triplet["label"]))

trainingTensor = np.array(trainingTensor)
labels = np.array(labels)
print("Training tensor has been built successfully")


print("Building testing tensor with labels...")
testTensor = []
numTestTriplets = len(testTriplets)

for k in tqdm(range(numTestTriplets)):
    # Select the kth triplet in the test triplet tensor
    triplet = testTriplets.iloc[k]

    # Get the features of all three images in the triplet and concatenate them
    reference = feature_saved[triplet["reference"]]
    positive = feature_saved[triplet["positive"]]
    negative = feature_saved[triplet["negative"]]
    triplet_tensor = np.concatenate((reference, positive, negative), axis=-1)

    # Add the resulting triplet of features to the test tensor 
    testTensor.append(triplet_tensor)

testTensor = np.array(testTensor)
print("Testing tensor has been built successfully")

# # 4) Create a neural net that takes the feature representations of the three images as inputs and is trained as a classification
# # network to return 1 if the first one looks more similar to the second one and 0 otherwise

# Create a model and train it with the tensor of concatenated features and corresponding labels
print("Creating model...")
inputs = Input(trainingTensor.shape[1:])
x = tf.keras.layers.Activation('relu')(inputs)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(2000)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Dense(units=1000)(x)
x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(units=200)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(units=50)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(units=20)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(units=10)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(units=5)(x)
x = tf.keras.layers.Activation('relu')(x)
x = tf.keras.layers.Dense(1)(x)
output = tf.keras.layers.Activation('sigmoid')(x)

# Create the model with the specified layers
model = Model(inputs=inputs, outputs=output)

# Compile the created model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Model created successfully")

# Train the model with the training data tensor
print("Training model...")
model.fit(x = trainingTensor, y = labels, epochs=10)
print("Training completed!")

# Apply the model to predict the classification output on the test tensor and threshold the result to be binary
print("Computing predictions for test data...")
output = model.predict(testTensor)
result = []
for i in range(len(output)):
    if (output[i] > 0.5):
        result.append(1)
    else:
        result.append(0)
np.savetxt(outputFile, result, fmt='%d')
print("Predictions have been computed and saved successfully")
