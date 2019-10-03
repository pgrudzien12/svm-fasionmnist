import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

label_names = ['T-shirt/top', 'Trouser', 'Pullover', 
    'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 
    'Bag', 'Ankle boot']

# Initialize SVM with parameters
def svmInit(C, gamma):
  model = cv2.ml.SVM_create()
  model.setGamma(gamma)
  model.setC(C)
  model.setKernel(cv2.ml.SVM_LINEAR)
  model.setType(cv2.ml.SVM_C_SVC)
  model.setTermCriteria((cv2.TERM_CRITERIA_EPS + 
                         cv2.TERM_CRITERIA_MAX_ITER, 
                         1000, 1e-3))
  return model


# Train SVM on data and labels
def svmTrain(model, samples, labels):
  model.train(samples, cv2.ml.ROW_SAMPLE, labels)


# predict labels for given samples
def svmPredict(model, samples):
  return model.predict(samples)[1]


# evaluate a model by comparing
# predicted labels and ground truth
def svmEvaluate(model, samples, labels):
  labels = labels[:, np.newaxis]
  pred = model.predict(samples)[1]
  correct = np.sum((labels == pred))
  err = (labels != pred).mean()
  print('label -- 1:{}, -1:{}'.format(np.sum(pred == 1), 
          np.sum(pred == -1)))
  return correct, err * 100


# create a directory if it doesn't exist
def createDir(folder):
  try:
    os.makedirs(folder)
  except OSError:
    print('{}: already exists'.format(folder))
  except Exception as e:
    print(e)

createDir('./models')

# compute HOG features for given images
def computeHOG(hog, images):
  hogFeatures = []
  for image in images:
    hogFeature = hog.compute(image)
    hogFeatures.append(hogFeature)
  return hogFeatures


# Convert HOG descriptors to format recognized by SVM
def prepareData(hogFeatures):
  featureVectorLength = len(hogFeatures[0])
  data = np.float32(hogFeatures).reshape(-1, featureVectorLength)
  return data

# Initialize HOG parameters
winSize = (32, 32) # size of the window over the image
blockSize = (8, 8) #  block noramlization step, oryginally 8x8
blockStride = (4, 4) # block noramlization, the stride of the block
cellSize = (4, 4) # ? size of the cell to compute histogram
nbins = 9 # number of bins in the histogram
derivAperture = 1
winSigma = -1
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = True
nlevels = 64
signedGradient = False

# Initialize HOG
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride,
                      cellSize, nbins,derivAperture,
                      winSigma, histogramNormType, L2HysThreshold, 
                      gammaCorrection, nlevels,signedGradient)

# Flags to turn on/off training or testing
trainModel = False
testModel = True


# ================================ Train Model =====================
if trainModel:
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images_bordered = []
    for i in range(0, len(training_images)):
        bordered = cv2.copyMakeBorder(training_images[i], 2,2,2,2,cv2.BORDER_CONSTANT, value=0)
        training_images_bordered.append(cv2.merge((bordered, bordered, bordered)))
    
    # Now compute HOG features for training images and 
    # convert HOG descriptors to data format recognized by SVM.
    # Compute HOG features for images
    hogTrain = computeHOG(hog, training_images_bordered)

    # Convert hog features into data format recognized by SVM
    trainData = prepareData(hogTrain)

    # Check dimensions of data and labels
    print('trainData: {}, trainLabels:{}'
            .format(training_images.shape, training_labels.shape))
    # Finally create an SVM object, train the model and save it.
    # Initialize SVM object
    model = svmInit(C=0.01, gamma=0)
    svmTrain(model, trainData, np.int32(training_labels))
    model.save('./models/fasion.yml')

# ================================ Test Model ===============
if testModel:
    # Load model from saved file
    model = cv2.ml.SVM_load('./models/fasion.yml')
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    test_images_bordered = []
    for i in range(0, len(test_images)):
        bordered = cv2.copyMakeBorder(test_images[i], 2,2,2,2,cv2.BORDER_CONSTANT, value=0)
        test_images_bordered.append(cv2.merge((bordered, bordered, bordered)))
    test_images_bordered = np.array(test_images_bordered)
    for i in range(0,10):
        testPosImages = test_images_bordered[test_labels == i]
        testPosLabels = test_labels[test_labels == i]

        hogPosTest = computeHOG(hog, np.array(testPosImages))
        testPosData = prepareData(hogPosTest)
    
        posCorrect, posError = svmEvaluate(model, testPosData, 
                                        np.array(testPosLabels))

        tp = posCorrect
        fp = len(testPosLabels) - posCorrect
        print('Category: {}, Correct: {}, Failed: {}, Total: {}, error: {}'
                .format(label_names[i], tp, fp, len(testPosLabels), posError))


    testPosImages = test_images_bordered
    testPosLabels = test_labels

    hogPosTest = computeHOG(hog, testPosImages)
    testPosData = prepareData(hogPosTest)

    posCorrect, posError = svmEvaluate(model, testPosData, 
                                    np.array(testPosLabels))

    tp = posCorrect
    fp = len(testPosLabels) - posCorrect
    print('Category: {}, Correct: {}, Failed: {}, Total: {}, error: {}'
            .format('All', tp, fp, len(testPosLabels), posError))


