""" SCIKIT-LEARN """
import time
import joblib

from datetime import timedelta

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

""" I have imported the necessary libraries for this project."""

start_time = time.time()

""" Variable start_time created to see how long the Neural Network will train for."""

MNIST = fetch_openml('mnist_784', parser="auto")

""" Got the needed data, the famous MNIST data set, containing thousands of already preprocessed images."""

X_train, X_test, Y_train, Y_test = \
    train_test_split(MNIST.data, MNIST.target, test_size=0.25)

""" 
    Split the data so I can train and test our Neural Network;
    I have divided the data set into four variables, for training and testing,
    to see how the model will learn from the presented images and comparing the two
    sets against each other to obtain an accuracy score; we have reshaped the data 
    so it can fit into the model; I have set a batch testing size of 25%, 
    with the rest being dedicated to training;
"""

MLP = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=300, activation='relu', solver='adam')

""" 
    The actual Neural Network; in this project I used a MultiLayer Perceptron, also known as an 'MLP'
    I have created three hidden layers, with 100 units each to maximize the accuracy of the model,
    and also to speed up the process of training, with a maximum of 300 iterations, as I supposed that
    I would get a pretty big accuracy out of this model, something which proved to be true;
    I am using an activation method called 'ReLU', or Rectified Linear Unit, as it pretty easy to
    train a model with it and also often achieves a better performance; I have also use and a solver 
    called 'adam' which seems to be of the best optimization algorithms as of right now,
    with my limited knowledge of Machine Learning.
"""


MLP.fit(X_train, Y_train)

"""Fitting the data on the model for the actual training process."""

file = 'Resources\\Finalized_Model'
joblib.dump(MLP, file)

""" Saving the trained model with the weights so I don't have to train it each time I want to use it."""

stop_time = time.time()-start_time
print(" Execution time: %s " % timedelta(seconds=round(stop_time)))

""" 
    The program has reached the end, so I am stopping the running time and calculating the difference
    between the starting time and ending time to see how long the project has been running for;
    in my latest run the project has run for about 3 minutes.
"""
