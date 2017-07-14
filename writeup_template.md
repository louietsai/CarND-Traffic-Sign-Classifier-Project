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
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


---


You're reading it! and here is a link to my [project code](https://github.com/louietsai/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

#Data Set Summary & Exploration


Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43

#2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a distribution diagram showing how the data is not well-balanced distributed.

![alt text][image1]

###Design and Test a Model Architecture

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
#Validation results

My final model results were:

| Augment    | pre-processing | Learning Rate	| Drop Out	| Epoch | Validation Accuracy	| Test Accuracy	| 
|:----------:|:--------------:|:-------------:| :-------:|:-----:|:-------------------:|:-------------:|
| image flip | hist and norm  | 0.001         |  0.5     |  80   |     0.931           |    0.908      |


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


