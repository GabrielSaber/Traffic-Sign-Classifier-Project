#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./traffic_sign_jpg/0.jpg "Traffic Sign 1"
[image5]: ./traffic_sign_jpg/1.jpg "Traffic Sign 2"
[image6]: ./traffic_sign_jpg/2.jpg "Traffic Sign 3"
[image7]: ./traffic_sign_jpg/3.jpg "Traffic Sign 4"
[image8]: ./traffic_sign_jpg/4.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is X_train.shape[0]
* The size of the validation set is X_valid.shape[0]
* The size of test set is X_test.shape[0]
* The shape of a traffic sign image is X_train[0].shape
* The number of unique classes/labels in the data set is len(np.unique(y_train))

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because
1- in case of traffic sign classifier the color has no effect on sing class, like the numbers classifier the color has no effect on number class, color one is the same as gray scale one
2- to reduce jpg depth this will affect the classifier performance and accuracy   

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because the data set has a zero mean and equal variance 

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used
lenet architecture with Adam Optimizer with learning_rate = 0.0008
epochs = 40 and batch size = 128

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.962
* validation set accuracy of 0.945 
* test set accuracy of 0.818

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* Lenet architecture
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* I have added two dropouts one after second convolution layer and one before last fully connected layer 
* Which parameters were tuned? How were they adjusted and why?
* i have decreased the learning rate for more accuracy and on the other side I have increased the EPOCHS I choose the batch size of 128 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
* Dropout is a regularization technique for reducing overfitting
* the dropout make the network don't rely on any activation so it will learn a redundant representation for everything 

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

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


The model was able to correctly guess 9 of the 11 traffic signs, which gives an accuracy of 81%. This compares favorably to the accuracy on the test set of 94.5%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a speed limit 20 sign (probability of 0.69), and the image does contain a stop sign. The top five soft max probabilities were

	| Probability         	|     Prediction	        					| 
	|:---------------------:|:---------------------------------------------:| 
	| .69         			| speed limit 20   								| 
	| .30     				| speed limit 30								|
	| .000017				| speed limit 70								|
	| .0000028	      		| speed limit 120					 			|
	| .00000004				| label 38      							    |


For the second image 
see the print top_k output  

	TopKV2(values=array
	([[  6.98079407e-01,   3.01900059e-01,   1.76116209e-05,
          2.85242413e-06,   4.40242189e-08],
       [  9.99987006e-01,   1.30184462e-05,   8.40866310e-09,
          5.52922291e-11,   4.64692304e-12],
       [  9.54180777e-01,   1.53640425e-02,   1.34247784e-02,
          4.15476877e-03,   3.90351471e-03],
       [  9.92799878e-01,   4.02353983e-03,   3.16339382e-03,
          1.29295395e-05,   1.98128845e-07],
       [  9.98143196e-01,   1.44927541e-03,   4.07388812e-04,
          6.50704592e-08,   1.40291589e-09],
       [  6.18424118e-01,   2.70140052e-01,   8.80665630e-02,
          2.12539490e-02,   7.34774629e-04],
       [  9.99996901e-01,   3.12340330e-06,   2.07612461e-09,
          6.22150179e-11,   1.59891440e-11],
       [  7.26213574e-01,   2.15222806e-01,   3.26035470e-02,
          1.77062433e-02,   5.90300467e-03],
       [  9.92805302e-01,   5.19173266e-03,   7.64373865e-04,
          4.15011338e-04,   4.13076312e-04],
       [  9.99996781e-01,   1.13239048e-06,   1.05382333e-06,
          1.03008915e-06,   3.53726435e-08],
       [  1.00000000e+00,   1.16864702e-13,   6.49779167e-14,
          4.40090469e-14,   2.91152905e-14]], dtype=float32), 
		
	  indices=array(
	  [[ 0,  1,  4,  8, 40],
       [ 1,  0,  4,  2, 38],
       [10,  5,  7, 19,  9],
       [ 3,  5,  2, 38,  1],
       [ 4,  0,  1,  8, 18],
       [ 1,  0, 40,  8,  6],
       [ 8,  7,  4,  0,  1],
       [14,  5,  2,  3, 12],
       [17,  9, 14, 41, 40],
       [34, 38, 13, 32, 35],
       [35, 20, 25, 40,  3]]))


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


