import imutils
import cv2
import numpy as np

# Function to resize an image to a deired widht and height
def resize_to_fit(image, width, height):
    # Dimensions of the image
    (h, w) = image.shape[:2]

    # If width is greater than height then resize width
    if w > h:
        image = imutils.resize(image, width=width)
    # Else, resize height
    else:
        image = imutils.resize(image, height=height)

    # Padding values for width and height to obtain disered dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # Pad the image and resize
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

# Function to train the LVQ-1 model
def train_lvq(epochs, vectors, train_data, learning_rate, labels):
  for epoch in range(epochs):
    for i, current in enumerate(train_data):
      # Calculate distances
      distances = np.linalg.norm(vectors - current, axis=1)
      # Winner neuron
      winner = np.argmin(distances)

      # Update vectors: If current label matches winner, get closer
      if labels[i] == winner:
        vectors[winner] += learning_rate * (current - vectors[winner])
      # Else, get further
      else:
        vectors[winner] -= learning_rate * (current - vectors[winner])
  return vectors

# Function to make predictions with the LVQ-1 model
def predict_lvq(X_test, final_vectors):
    Y_pred = np.array([])

    for i, entrada in enumerate(X_test):
      distances = np.linalg.norm(final_vectors - entrada, axis=1)
      ganadora = np.argmin(distances)
      Y_pred = np.append(Y_pred, ganadora)
        
    Y_pred = Y_pred.astype(int)

    return Y_pred

# Function to get random indexes of data from the total classes
def random_indexes(num_classes, Y_train):
    classes = [] # Store the indexes

    # Save a random index for a point of data from each class.
    for i in range(num_classes):
      random_index = np.random.choice(np.where(Y_train == i)[0])
      classes.append(random_index)
    
    return classes