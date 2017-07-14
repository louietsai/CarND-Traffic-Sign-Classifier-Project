#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/data_visualize.png "data visualize"
[image2]: ./examples/raw.png "raw image"
[image3]: ./examples/pre-processimg.png "pre-processing"
[image4]: ./traffic-signs-data/private/0-1.jpg "Traffic Sign 1"
[image5]: ./traffic-signs-data/private/1-1.jpg "Traffic Sign 2"
[image6]: ./traffic-signs-data/private/2-1.jpg "Traffic Sign 3"
[image7]: ./traffic-signs-data/private/3-1.jpg "Traffic Sign 4"
[image8]: ./traffic-signs-data/private/40-1.jpg "Traffic Sign 5"
[image9]: ./traffic-signs-data/private/25-1.jpg "Traffic Sign 6"
[image10]: ./traffic-signs-data/private/36-1.jpg "Traffic Sign 7"
[image11]: ./traffic-signs-data/private/7-1.jpg "Traffic Sign 8"
[image12]: ./traffic-signs-data/private/38-1.jpg "Traffic Sign 9"
[image13]: ./traffic-signs-data/private/9-1.jpg "Traffic Sign 10"

---


You're reading it! and here is a link to my [project code](https://github.com/louietsai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

**Data Set Summary & Exploration


Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a distribution diagram showing how the data is not well-balanced distributed.

![alt text][image1]

**Design and Test a Model Architecture

due to none well-balanced distributed dataset, I decided to generate additional data.
I used image flip to expand the data number from 34799 to 59788.
Becuase the brightness may not be good for some data images.
Therefore I use OpenCV to equalize the histogram of the Y channel, and used OpenCV to normalize the image.
finally I shuffle training data.
The difference between the original data set and the augmented data set is the following ... 
here is an example for raw image data.
![alt text][image2]
here is an exmple after pre-processing
![alt text][image3]





My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  Input = 28x28x6. Output = 14x14x6	|
| Convolution 3x3	    | 1x1 stride Output = 10x10x16									|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Input = 10x10x16. Output = 5x5x16 |
| Flatten |  Input = 5x5x16. Output = 400
| Fully connected		| Input = 400. Output = 120       									|
| RELU					|												|
| dropout | 50% rate |
| Fully connected		| Input = 120. Output = 84 |
| RELU					|												|
| dropout | 50% rate |
| Fully connected		| Input = 84. Output = 43 |

 


To train the model, I used an tf.train.AdamOptimizer with 128 batch size, and I run 80 EPOCHS with 0.001 learning rate for the model.
01
#**Validation results

My final model results were:

| Augment    | pre-processing | Learning Rate	| Drop Out	| Epoch | Validation Accuracy	| Test Accuracy	| 
|:----------:|:--------------:|:-------------:| :-------:|:-----:|:-------------------:|:-------------:|
| image flip | hist and norm  | 0.001         |  0.5     |  80   |     0.931           |    0.908      |

If I don't adopt 2 drop out layers, the model seems to be overfitted, and the validation accuracy will be ~0.90.


 

**Test a Model on New Images



Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]![alt text][image9] 
![alt text][image10] ![alt text][image11]![alt text][image12] 
![alt text][image13]



** New Image Accuracy

Test Accuracy = 0.908
My Images Accuracy = 0.722

|0 : Label= 3 ,Predicted= 2 = 1.0|
|1 : Label= 37 ,Predicted= 37 = 1.0|
|2 : Label= 7 ,Predicted= 7 = 1.0|
|3 : Label= 19 ,Predicted= 19 = 1.0|
|4 : Label= 9 ,Predicted= 9 = 1.0|
|5 : Label= 12 ,Predicted= 12 = 1.0|
|6 : Label= 18 ,Predicted= 18 = 1.0|
|7 : Label= 25 ,Predicted= 14 = 1.0|
|8 : Label= 13 ,Predicted= 13 = 1.0|
|9 : Label= 32 ,Predicted= 32 = 1.0|
|10 : Label= 0 ,Predicted= 0 = 1.0|
|11 : Label= 40 ,Predicted= 40 = 1.0|
|12 : Label= 36 ,Predicted= 18 = 0.999983|
|13 : Label= 1 ,Predicted= 9 = 1.0|
|14 : Label= 14 ,Predicted= 14 = 1.0|
|15 : Label= 2 ,Predicted= 1 = 1.0|
|16 : Label= 38 ,Predicted= 38 = 1.0|
|17 : Label= 17 ,Predicted= 17 = 1.0|
Counted 5 bad predictions



The model was able to correctly guess 12 of the 17 traffic signs, which gives an accuracy of 72%. 





