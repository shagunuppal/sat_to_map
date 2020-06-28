# Sat2Map: Generating Maps from Satellite Images

Learning mappings from one image to another has been a problem of great interest since recent times in the field of image analysis, computer vision and learning models. With this project, we aim to develop an image-to-image learning model, which will be able to synthesise a given image representation to any other target scene representation.  <br>
To address this problem, we are specifically focussing on generating maps from the corresponding satellite images for cities. Such kind of a problem seeks great attention as well as application in real life situations for the purpose of navigation and generating city maps. The dataset to be used for the same consists of satellite images of cities along with their corresponding generated maps as the ground truth. 

### Dataset Details 
The dataset containing satellite image of cities obtained via web scraping from Google Maps was available publically provided by UC Berkeley’s Electrical and Computer Science Dept. The data is available as a concatenated form of satellite and map images. <br>
The details of the dataset are as follows: 
* Number images in dataset (train + validation): 2194 
* Size of each combined image (the satellite and the map image) in dataset: 1200 x 600 so that means each image (map or satellite image) is of size 600 x 600. 
* Size of complete dataset: 257.7 MB <br>
Additionally, we also test the approach on web scrapped images from our locality.

<a href="http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/">Link to Dataset</a>

### Learning Techniques 
We intend to compare the performance on the task of map generation from a given satellite image on a trivial neural network as well as simple and compound generative models that include the following: 
* Convolutional Neural Network (CNN): This is a simple neural network consisting of convolution layers. CNNs learn on the basis of learning the loss function to enhance the quality of the results, the loss function being manually defined as per the task taken into consideration. 
* Variational Autoencoder (VAE): This is an autoencoder consisting of an encoder and decoder and a gaussian prior imposed on the latent space. The training approach is based on reducing the reconstruction loss between the predicted and ground truth images. 
* Generative Adversarial Network (GAN): This is a complex model consisting of a generator capable of producing fake data samples (images in our case) and a discriminator whose job is to differentiate between the real and the fake samples. The learning is influenced by the min-max tradeoff between the generator and the discriminator. 

### Evaluation Metrics 
In order to estimate an evaluation metric for all the three CNN, VAE and GAN based approaches, we consider taking the euclidean pixel to pixel difference between the ground truth and the predicted image as the desired measure. However, for the GAN based approach, the idea is not only to get the predicted synthesised image closer to the ground truth image but also at the same time generated the predicted image no far from reality. 
<br>
In order to cover up this high-level goal of “real v/s fake” to make the output more genuine to human vision and observation, we aim to conduct a user survey. For each person appearing for the survey, he is shown a set of images, each for a time duration of 1 second and then given enough time to classify it as a real or a fake image. After certain such trials, he is given some feedback accordingly. When evaluating the final performance, the evaluator is given real and fake images corresponding to different set of inputs in order to maintain the complication of the task.
<br>
Both these evaluation metrics help us in finally commenting on the performance of various techniques and algorithms implemented in order to tackle the problem of consideration.

