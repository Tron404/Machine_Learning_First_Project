import numpy as np
import matplotlib.pyplot as plt
import time
import joblib

from datetime import timedelta

from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

""" I have imported the necessary libraries to test the model I created and trained in the MLP.py file."""

start_time = time.time()

""" Variable start_time created to see how long the Neural Network will train for."""

MNIST = fetch_openml('mnist_784', parser="auto")
X_train, X_test, Y_train, Y_test =\
    train_test_split(MNIST.data, MNIST.target, test_size=0.9)

""" 
    This data that I am splitting is the same as the one before, but this time I am focusing more on testing
    the actual Neural Network, that's why I have a test_size of 90%; in all honesty this could result in the model
    having a big accuracy not to being properly trained, but to being able to recognise some of the models it may
    have encountered in the training process.
"""

Trained_MLP = joblib.load('Resources\\Finalized_Model')

""" Loading the trained model I previously saved."""

predictions = Trained_MLP.predict(X_test)

""" Actually predicting the labels for the input images."""

score_init = Trained_MLP.score(X_test, Y_test)
score_readable = round(float(score_init), 2)*100
print(str(score_readable) + "%")

""" Obtaining the accuracy score and printing it."""

cm = metrics.confusion_matrix(Y_test, predictions)

print(cm)

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
