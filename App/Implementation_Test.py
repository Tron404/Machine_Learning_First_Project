import numpy as np
import joblib
import cv2

""" 
    I have imported the necessary libraries to test the model on my own images, 
    in this case one by one.
"""

Trained_MLP = joblib.load('Resources\\Finalized_Model')

""" Loading the trained model I previously saved."""

img = cv2.imread('Processed_Images\\blur.jpg')

""" 
    For privacy reasons some of the path has been removed, the <NAME> part can be
    replaced with the name of another user or even the entire path can be replaced.
"""
""" Loading my own image into the project"""

kernel = np.ones((5, 5), np.uint8)

""" 
    Creating a kernel that consists of a 5x5 matrix filled with ones, all of which 
    are of unsigned intenger type, meaning from 0 to 255. Its usage will be explained
    down. 
"""

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""
    Converting our image to a gray one, meaning conversion to grayscale, as for this 
    project we don't need RGB values, but just one, which represents the intensity of
    one pixel on a scale from 0 to 255. 
"""

blur = cv2.GaussianBlur(gray, (5, 5), 0)

"""
    Blurring( or smoothing) our image to remove any potential noise which could
    interfere with our prediction by using a tuplet of (5, 5) as our kernel, which
    IS different from the kernel variable I used in a function below.
"""

ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

"""
    The threshold at which pixels will be converted, in this case, to black and white, 
    because of the THRESH_BINARY_INV flag, which also inverses the colours to 
    the ones mentioned above; the THRESH_OTSU flag is used to determine the the threshold 
    value by calculating a measure of spread for the pixel levels on each side of the threshold,
    which we add to the previous flag to obtain our new image and which yields a better result 
    for our model as to just using the THRESH_BINARY_INV flag. The minimum value for the threshold 
    is 100, with a maximum of 255. We don't need the returned threshold value found in the ret variable.
"""

morphology = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

"""
    Here we are using some morphological operations, meaning erosion and dilation,
    to improve the quality of image for the model; here I am "closing" some
     of the shapes in the image, as they might have holes which could interfere
    with the prediction of the model; here is where I am using the aforementioned
    kernel, so the cv2.morphology function knows what operation to do on the image.
"""

contours, hierarchy = cv2.findContours(morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

"""
    Finding the contours in the image using the "thresh" image with the RETR_EXTERNAL
    flag, which returns only the outermost contours so the image won't have contours
    inside each other which could lead to messy detections; the CHAIN_APROX_SIMPLE 
    removes all redundant points in the contour, leaving only the essential part. 
    We don't need the hierarchy of the contours found in the hierarchy variable.
"""

"""
    Saving the images as I wanted to see the processing being done to understand 
    the whole process and the functions better.
"""

for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    """Creating the actual contours."""
    digit = morphology[y:y+h, x:x+w]
    """Extracting each found contour."""
    resized_digit = cv2.resize(digit, (18, 18))
    pad_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
    """Resizing and padding the digits for a better prediction."""
    prediction = Trained_MLP.predict(pad_digit.reshape(1, 28 * 28))
    """The actual prediction."""
    probability = float(Trained_MLP.predict_proba(pad_digit.reshape(1, 28 * 28))[:, int(prediction)])
    """The probability of each prediction put in a readable format."""
    text = str(int(prediction)) + "-" + str(round(probability, 2) * 100) + "%"
    """The prediction and the probability combined."""
    cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=5)
    cv2.putText(img, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    """
        Drawing rectangles around the contours to show the found digits and also 
        putting text to understand what the model found.
    """

cv2.imshow("Implementation_Test", img)
cv2.waitKey(0)

"""Showing the image with the predicted labels"""
