import tensorflow as tf 
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np  
import os

#Get the MNIST sample data and split it into Tuples
mnist = tf.keras.datasets.mnist
(xTrain, yTrain), (xTest, yTest) = mnist.load_data()

#Normalize data, in other words make its' length 1 scaling it down to make it easier to compute
#data points range from 0 to 1. 
xTrain = tf.keras.utils.normalize(xTrain, axis = 1)
xTest = tf.keras.utils.normalize(xTest, axis = 1)

#Define neural network, add input layer add 2 hidden layers and an output layer.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))             #Flatten = 1D -layer. Essentially the input -layer
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu)) #Dense = the neurons are all connected to the previous as well as next layer
model.add(tf.keras.layers.Dense(units = 128, activation = tf.nn.relu)) #Units = amount of neurons = the more sophisticated the layer. 
model.add(tf.keras.layers.Dense(units = 10, activation = tf.nn.softmax)) #Output -layer. units = 10 for the 10 digits. Softmax takes all the outputs, and scales the values to one value 
                                                                           #which represents the probability of the number being the result of the classification. 

#Compile and optimize model. 
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#Train the model. epochs = how many times the same data is shown to the model
model.fit(xTrain, yTrain, epochs = 3)

#Evaluate the model
loss, accuracy = model.evaluate(xTest, yTest)
print(f'Model accuracy: {accuracy}')
print(f'Model loss: {loss}')

#Save your fresh, newly trained model
#this piece of code is commented out so we don't just keep on creating models each time we run the script.
#model.save('handwrittenDigitsModel')

#for some unfortunate reason, loading the model gives a keyword argument error. Why, i don't know.
#model = tf.keras.models.load_model('ml/numberRecognition/handwrittenDigitsModel') 

#Load the custom images
#feel free to use your own as long as they're 28x28 px

imgNumb = 1
while os.path.isfile(f'ml/numberRecognition/digits.model/digits/digit{imgNumb}.png'):

    img = cv.imread(f'ml/numberRecognition/digits.model/digits/digit{imgNumb}.png')[:, :, 0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The number is most probably: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap = plt.cm.binary)
    plt.show()
    imgNumb += 1