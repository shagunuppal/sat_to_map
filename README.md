# Sat_to_Map

<h1>Motivation of the problem statement</h1>
<br>
Learning mappings from one image to another has been a problem of great interest since recent times in the field of image analysis, computer vision and learning models. With this project, we aim to develop an image-to-image learning model, which will be able to synthesise a given image representation to any other target scene representation.  
<br><br>
To address this problem, we are specifically focussing on generating maps from the corresponding satellite images for cities. Such kind of a problem seeks great attention as well as application in real life situations for the purpose of navigation and generating city maps. The dataset to be used for the same consists of satellite images of cities along with their corresponding generated maps as the ground truth. <br>

<h1> Data Acquisition </h1>
<br>
The dataset containing satellite image of cities obtained via web scraping from Google Maps was available publically (http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/) provided by UC Berkeley’s Electrical and Computer Science Dept. The data is available as a concatenated form of satellite and map images. <br>
The details of the dataset are as follows: <br>
Number images in dataset (train + validation): 2194 <br>
Size of each combined image (the satellite and the map image) in dataset: 1200 x 600 so that means each image (map or satellite image) is of size 600 x 600. >br>
Size of complete dataset: 257.7 MB
<br>

<h1> Preprocessing Techniques </h1>
<br>
The fact that the images are coupled together, requires us to seperate the two images of each scene by simply cropping to retrieve the satellite image as our training datapoint and the corresponding map images becomes the ground truth data for the same, required for computing the loss between our predicted and the desired images.
<br>

<h1> Learning Techniques </h1>
<br>
We intend to compare the performance on the task of map generation from a given satellite image on a trivial neural network as well as simple and compound generative models that include the following: <br>
<b>Convolutional Neural Network (CNN) : </b> This is a simple neural network consisting of convolution layers. CNNs learn on the basis of learning the loss function to enhance the quality of the results, the loss function being manually defined as per the task taken into consideration. <br>
<b> Variational Autoencoder (VAE) : </b> This is an autoencoder consisting of an encoder and decoder and a gaussian prior imposed on the latent space. The training approach is based on reducing the reconstruction loss between the predicted and ground truth images. <br>
<b> Generative Adversarial Network (GAN) : </b> This is a complex model consisting of a generator capable of producing fake data samples (images in our case) and a discriminator whose job is to differentiate between the real and the fake samples. The learning is influenced by the min-max tradeoff between the generator and the discriminator. <br>
<br>

<h1> Startegy for Model Selection </h1>
<br>
We select kernel based approach to use convolution filters. Also, in the neural network architecture, both linear and non-linear approaches are also embedded within in the form of fully connected components as well as activation layers while training models like the VAE. <br>
In order to tune the hyperparameters like learning rate and gradient descent approaches, we use the holdout technique. Based on the different values of the hyperparameters and hence, different configurations of the model, we compare the different evaluation metrics and choose the one which guarantees the highest performance measure. 
<br>

<h1> Training Approaches </h1>
<br>
For the task of training a VAE and a CNN, we propose to use a Stochastic Gradient Descent which is relatively faster than the Batch Gradient Descent. <br>
We intend to explore and compare the performance of two different types of gradient descent approaches - Batch GD and Stochastic GD for training our models on GANs. We iteratively alternate between one gradient descent step on the discriminator to one on the generator so as to minimise the loss between the ground truth image and the output image of the network. We intend to use Adam’s optimiser for mini-batch gradient descent with appropriate learning rates and momentum parameters.
<br>

<h1> Evaluation Metrics </h1>
<br>
In order to estimate an evaluation metric for all the three CNN, VAE and GAN based approaches, we consider taking the euclidean pixel to pixel difference between the ground truth and the predicted image as the desired measure. However, for the GAN based approach, the idea is not only to get the predicted synthesised image closer to the ground truth image but also at the same time generated the predicted image no far from reality. <br> 
So, in order to cover up this high-level goal of “real v/s fake” to make the output more genuine to human vision and observation, we aim to conduct a user survey. For each person appearing for the survey, he is shown a set of images, each for a time duration of 1 second and then given enough time to classify it as a real or a fake image. After certain such trials, he is given some feedback accordingly. When evaluating the final performance, the evaluator is given real and fake images corresponding to different set of inputs in order to maintain the complication of the task.
<br><br>
Both these evaluation metrics help us in finally commenting on the performance of various techniques and algorithms implemented in order to tackle the problem of consideration.
<br>

