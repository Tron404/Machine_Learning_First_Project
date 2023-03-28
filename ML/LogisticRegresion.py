""" SCIKIT-LEARN """
import numpy as np
import matplotlib.pyplot as plt
import time

from datetime import timedelta

from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

""" I have imported the necessary libraries for this project."""

start_time = time.time()

""" Variable start_time created to see how long the Neural Network will train for."""

MNIST = fetch_openml('mnist_784', parser="auto")

""" Got the needed data, the famous MNIST data set, containing thousands of already preprocessed images."""

X_train, X_test, Y_train, Y_test = train_test_split(MNIST.data, MNIST.target, test_size=0.25)

""" 
    Split the data so I can train and test our Neural Network;
    I have divided the data set into four variables, for training and testing,
    to see how the model will learn from the presented images and comparing the two
    sets against each other to obtain an accuracy score; we have reshaped the data 
    so it can fit into the model; I have set a batch testing size of 25%, 
    with the rest being dedicated to training;
"""

LogisticRegr = LogisticRegression(solver='saga', max_iter=300)

"""
    A logistic regression algorithm. My first actual try at Machine Learning.
    Solver "saga" which seems to be the best for large amount of data, with a
    maximum of 10000 iterations.
"""

LogisticRegr.fit(X_train, Y_train, sample_weight=0.1)

"""Fitting the data on the model for the actual training process."""

"""May take A LONG time."""

predictions = LogisticRegr.predict(X_test)

""" Actually predicting the labels for the input images."""

score_init = LogisticRegr.score(X_test, Y_test)
score_readable = round(float(score_init), 2)*100
print(str(score_readable) + "%")

""" Obtaining the accuracy score and printing it."""

cm = metrics.confusion_matrix(Y_test, predictions)

""" 
    Creating a Confusion Matrix in the form of a Correlogram
    to better visualise what the model got wrong or right 
    for better analysis of the situation.
"""

plt.figure(figsize=(9, 9))
plt.imshow(cm, interpolation='nearest', cmap='Set1')
plt.title('Confusion Matrix', size=15)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
plt.yticks(tick_marks, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], size=10)
plt.tight_layout()
plt.ylabel('Actual label', size=15)
plt.xlabel('Predicted label', size=15)
width, height = cm.shape

""" 
    I can customise the Correlogram pretty much how I want,
    like the name of the created window and the colour scheme
    and also the size.
"""

"""
    In case the plot breaks.

    bottom, top = plt.ylim()
    bottom += 0.5
    top -= 0.5
    plt.ylim(bottom, top)

"""

stop_time = time.time()-start_time
print(" Execution time: %s " % timedelta(seconds=round(stop_time)))

""" 
    The program has reached the end, so I am stopping the running time and calculating a difference to see
    how long the project has been running for.
"""


for x in range(width):
    for y in range(height):
        plt.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')
plt.show()

""" Putting the actual values in the Matrix."""
