#**Behavioral Cloning** 

##Xiangjun Fan

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/center.jpg "Center"
[image2]: ./examples/right.jpg "Right"
[image3]: ./examples/left.jpg "Left"
[image4]: ./examples/center_flip.png "Center Flip"
[image5]: ./loss_hist.png "train valid loss plot"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* run1.mp4 for autonomous mode driving video
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. Simulator screen resolution 640x480, graphics quality Fastest.
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I chose LeNet as a starting point and then used a combined structure with 5 CNNs + 4 Dense layers. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. Dropout layers were used in network to avoid overfitting. Train and valid loss were monitored, and model checkpoint were saved. The model used was the one with the lowest valid loss. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Batch size and cropping size are tuned manually to achieve best validation loss.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to iteratively increase model complexity, add data augmemntation and tune hypeparameter to achieve a good validation loss. 

My first step was to use a convolution neural network model similar to LeNet. Date was fed in without normalization or augmementation. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. The first mse loss was not very good.

Then I added data normlization and image cropping. The model was also changed to be similar to 5 layer CNN structure in the course video. The CNN and dense layers parameters were kept as is. I chose to use batch SGD with batch size of 128. From training loss and validation loss monitoring, I used Keras checkpoint to save all good models(with low valid loss). The train loss v.s. valid loss plot was monitored to make sure the model used was not overfitting. The model I picked for testing was from the last epoch in the following plot.

![alt text][image5]



The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, to improve the driving behavior in these cases, I added more data, use all three camera views, and applied image flipping.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is as follows.

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |           160x320x3 RGB image            |
|     Lambda      |              Normlize data               |
| Convolution 5x5 | subsample (2,2), output channel 24, RELU |
| Convolution 5x5 | subsample (2,2), output channel 36, RELU |
| Convolution 5x5 | subsample (2,2), output channel 48, RELU |
|     Dropout     |              keep prob 0.5               |
| Convolution 3x3 | subsample (1,1), output channel 64, RELU |
| Convolution 3x3 | subsample (1,1), output channel 64, RELU |
|     Dropout     |              keep prob 0.5               |
|      Dense      |             output size 100              |
|     Dropout     |              keep prob 0.5               |
|      Dense      |              output size 50              |
|     Dropout     |              keep prob 0.5               |
|      Dense      |              output size 10              |
|      Dense      |              output size 1               |

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steering back to road once reaching the lane line. These images show what a recovery looks like starting from right to left :

![alt text][image2]

![alt text][image3]


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help train the model for making both left and right turns, while simulator mostly gave left turns (counter clockwise lap). For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image4]

After the collection process, I had 30078 number of data points. I then preprocessed this data by normalizing RGB channels, and cropping off top and bottom part of the image.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by that epoch 4 gave the lowest validation loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
