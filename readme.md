# Fasion MNIST Classification using Support Vector Machines

## How to use
1. Install tensorflow (used only to get the dataset), I used version 2.0
2. Use flags in fasion.py to train/test model

## Methodology
First the training set undergo the HoG feature detection. The output vector from the previous step is then applied to the SVM along with the labels.

Results are calculated based on the traning set that is also first transformed using HoG descriptor.

## Results

This classifier turns out to be quite good compared even compared to the NN's. Some classes like shoes and bags most likely have differentiating HOG descriptors and the SVM could separate those from the other classes quite well.

Here is the comparison:
https://github.com/zalandoresearch/fashion-mnist

The parameters can be improved to get even better results.

Total error on the test set is 10.09%

* Category: T-shirt/top, Correct: 820, Failed: 180, Total: 1000, error: 18.0
* Category: Trouser, Correct: 974, Failed: 26, Total: 1000, error: 2.6
* Category: Pullover, Correct: 868, Failed: 132, Total: 1000, error: 13.200000000000001
* Category: Dress, Correct: 922, Failed: 78, Total: 1000, error: 7.8
* Category: Coat, Correct: 841, Failed: 159, Total: 1000, error: 15.9
* Category: Sandal, Correct: 976, Failed: 24, Total: 1000, error: 2.4
* Category: Shirt, Correct: 664, Failed: 336, Total: 1000, error: 33.6
* Category: Sneaker, Correct: 981, Failed: 19, Total: 1000, error: 1.9
* Category: Bag, Correct: 984, Failed: 16, Total: 1000, error: 1.6
* Category: Ankle boot, Correct: 961, Failed: 39, Total: 1000, error: 3.9
* Category: All, Correct: 8991, Failed: 1009, Total: 10000, error: 10.09

# Credits
Some of the SVM code for learning was taken from the OpenCV course
https://opencv.org/courses/
