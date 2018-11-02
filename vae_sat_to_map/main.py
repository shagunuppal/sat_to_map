import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from random import randint
import numpy as np
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import os, time, sys
import matplotlib.pyplot as plt
import itertools
from utils import imshow_grid
from itertools import cycle
from torch.utils.data import DataLoader
from utils import transform_config, reparameterize
import argparse
import glob
import cv2

parser = argparse.ArgumentParser()

# add arguments
parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")

parser.add_argument('--batch_size', type=int, default=64, help="batch size for training")
parser.add_argument('--image_size', type=int, default=128, help="height and width of the image")
parser.add_argument('--initial_learning_rate', type=float, default=0.00001, help="starting learning rate")

parser.add_argument('--nv_dim', type=int, default=64, help="dimension of varying factor latent space")
parser.add_argument('--nc_dim', type=int, default=512, help="dimension of common factor latent space")
parser.add_argument('--class_to_remove', type=list, default=[], help="classes to remove from training dataset")
parser.add_argument('--num_classes', type=int, default=25, help="number of classes on which the data set trained")

# arguments to control per iteration training of architecture
parser.add_argument('--train_auto_encoder', type=bool, default=True, help="train the auto-encoder part")
parser.add_argument('--train_reverse_cycle', type=bool, default=True, help="train the reverse consistency cycle")
parser.add_argument('--train_decorr_classifier', type=bool, default=False, help="train classifier to decorrelate")

# loss function coefficient
# 3 reconstruction coef for 64 dim space
parser.add_argument('--reconstruction_coef', type=float, default=2., help="coefficient for reconstruction term")
parser.add_argument('--reverse_cycle_coef', type=float, default=10., help="coefficient for reverse cycle loss term")
parser.add_argument('--kl_divergence_coef', type=float, default=3., help="coefficient for KL-Divergence loss term")
parser.add_argument('--decorr_coef', type=float, default=0., help="coefficient for decorrelation loss term")

parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")


# paths to save models
parser.add_argument('--encoder_save', type=str, default='encoder', help="model save for encoder")
parser.add_argument('--decoder_save', type=str, default='decoder', help="model save for decoder")
parser.add_argument('--classifier_save', type=str, default='classifier', help="model save for nv classifier")

parser.add_argument('--log_file', type=str, default='log.txt', help="text file to save training logs")

parser.add_argument('--load_saved', type=bool, default=False, help="flag to indicate if a saved model will be loaded")
parser.add_argument('--start_epoch', type=int, default=0, help="flag to set the starting epoch for training")
parser.add_argument('--end_epoch', type=int, default=150, help="flag to indicate the final epoch of training")

FLAGS = parser.parse_args()

################################################################################################################################################################################

num_epochs = 500
batch_size = 8
learning_rate = 0.005

nc_dim = 160

img_transform = transforms.Compose([transforms.ToTensor()])

def to_img(x):
	x = x.clamp(0, 1)
	x = x.view(x.size(0), 3, 128, 128)
	return x

class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
	
		self.convolution_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True)
		self.convolution_1_in = nn.InstanceNorm2d(num_features=32)
		self.ELU_1 = nn.ReLU(inplace=True)

		self.convolution_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)
		self.convolution_2_in = nn.InstanceNorm2d(num_features=32)
		self.ELU_2 = nn.ReLU(inplace=True)

		self.convolution_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)
		self.convolution_3_in = nn.InstanceNorm2d(num_features=64)
		self.ELU_3 = nn.ReLU(inplace=True)

		self.convolution_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)
		self.convolution_4_in = nn.InstanceNorm2d(num_features=128)
		self.ELU_4 = nn.ReLU(inplace=True)

		self.convolution_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)
		self.convolution_5_in = nn.InstanceNorm2d(num_features=256)
		self.ELU_5 = nn.ReLU(inplace=True)

		self.common_factor_mu = nn.Linear(in_features=1024, out_features=nc_dim, bias=True)
		self.common_factor_logvar = nn.Linear(in_features=1024, out_features=nc_dim, bias=True)

		self.common_factor_dec = nn.Linear(in_features=nc_dim, out_features=2048, bias=True)

		self.relu = nn.ReLU(inplace=True)

		self.deconvolution_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)
		self.deconvolution_1_in = nn.InstanceNorm2d(num_features=256)
		self.ELU_1_ = nn.ReLU(inplace=True)

		self.deconvolution_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=0, bias=True)
		self.deconvolution_2_in = nn.InstanceNorm2d(num_features=128)
		self.ELU_2_ = nn.ReLU(inplace=True)

		self.deconvolution_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
		self.deconvolution_3_in = nn.InstanceNorm2d(num_features=64)
		self.ELU_3_ = nn.ReLU(inplace=True)

		self.deconvolution_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True)
		self.deconvolution_4_in = nn.InstanceNorm2d(num_features=32)
		self.ELU_4_ = nn.ReLU(inplace=True)

		self.deconvolution_5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True)
		self.sigmoid_final = nn.Sigmoid()


	def encode(self, x):
		x = self.ELU_1(self.convolution_1_in(self.convolution_1(x)))
		x = self.ELU_2(self.convolution_2_in(self.convolution_2(x)))
		x = self.ELU_3(self.convolution_3_in(self.convolution_3(x)))
		x = self.ELU_4(self.convolution_4_in(self.convolution_4(x)))
		x = self.ELU_5(self.convolution_5_in(self.convolution_5(x))) 
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		mu = self.common_factor_mu(x)
		sigma = self.common_factor_logvar(x)
		return mu, sigma

	def reparametrize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = torch.FloatTensor(std.size()).normal_().cuda()
		eps = Variable(eps)
		return eps.mul(std).add_(mu)

	def decode(self, z):
		x = self.relu(self.common_factor_dec(z))
		x = x.view(x.size(0), 512, 2, 2)
		x = self.ELU_1_(self.deconvolution_1_in(self.deconvolution_1(x)))
		x = self.ELU_2_(self.deconvolution_2_in(self.deconvolution_2(x)))
		x = self.ELU_3_(self.deconvolution_3_in(self.deconvolution_3(x)))
		x = self.ELU_4_(self.deconvolution_4_in(self.deconvolution_4(x)))
		x = self.sigmoid_final(self.deconvolution_5(x))
		return x

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparametrize(mu, logvar)
		res = self.decode(z)
		return res, mu, logvar


model = VAE().cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
reconstruction_function = nn.MSELoss(size_average=False)

def loss_function(recon_x, x, mu, logvar):
	"""
	recon_x: generating images
	x: origin images
	mu: latent mean
	logvar: latent log variance
	"""
	BCE = reconstruction_function(recon_x, x)  # mse loss
	KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
	KLD = torch.sum(KLD_element).mul_(-0.5)
	# KL divergence
	return BCE + KLD

def train(batchsize):
	print('Loading Sat2Map Dataset...')
	data_dir = '../processed_dataset/train/images/'
	label_dir = '../processed_dataset/train/maps1/'
	data_dir_1 = '../processed_dataset/val/images/'
	label_dir_1 = '../processed_dataset/val/maps1/'

	dset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
	train_data_dir = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)

	dset1 = datasets.ImageFolder(label_dir, transform=transforms.ToTensor())
	train_label_dir = torch.utils.data.DataLoader(dset1, batch_size=batch_size, shuffle=False)

	dset_more = datasets.ImageFolder(data_dir_1, transform=transforms.ToTensor())
	train_data_dir_more = torch.utils.data.DataLoader(dset_more, batch_size=batch_size, shuffle=False)

	dset_1_more = datasets.ImageFolder(label_dir_1, transform=transforms.ToTensor())
	train_label_dir_more = torch.utils.data.DataLoader(dset_1_more, batch_size=batch_size, shuffle=False)        

	train_images = []
	train_ground_truths = []
	for i, (img, _) in enumerate(train_data_dir):
		train_images.append(img)
	for i, (img, _) in enumerate(train_label_dir):
		train_ground_truths.append(img)

	for i, (img, _) in enumerate(train_data_dir_more):
		train_images.append(img)
	for i, (img, _) in enumerate(train_label_dir_more):
		train_ground_truths.append(img)
	
	train_ground_truths = train_ground_truths[:len(train_ground_truths)-2]
	train_images = train_images[:len(train_images)-2]

	len_train = len(train_images)

	for epoch in range(num_epochs):
		model.train().cuda()
		train_loss = 0
		for i in range(len(train_images)):
			img = Variable(train_images[i].cuda())
			map_img = Variable(train_ground_truths[i].cuda()) 
			optimizer.zero_grad()
			reconstructed, mu, sigma = model(img)
			loss = loss_function(reconstructed, map_img, mu, sigma)
			loss.backward()
			train_loss += loss.data[0]
			optimizer.step()
			if(i%100==0):
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch,
					i * len(img),
					len_train, 
					100. * i / len_train,
					loss.data[0] / len(img)))
		print('====> Epoch: {} Average loss: {:.4f}'.format(
			epoch, train_loss / len_train))
		if epoch % 10 == 0:
			ground_truth_img = to_img(img.cpu().data)
			ground_truth_map = to_img(map_img.cpu().data)
			rec = to_img(reconstructed.cpu().data)
			batch_img = torch.zeros(24, 3, 128, 128)
			batch_img[0:8, :, :, :] = ground_truth_img
			batch_img[8:16, :, :, :] = ground_truth_map
			batch_img[16:24, :, :, :] = rec
			save_image(batch_img, './vae_images/batch_images_{}.png'.format(epoch), nrow=8)
	return model

def load_model():
	model.load_state_dict(torch.load('./vae.pth'))
	return model

def save_model(model):
	torch.save(model.state_dict(), './vae.pth')

def generate_image(model):
	z = Variable(torch.FloatTensor(1,nc_dim).normal_(), requires_grad=True)
	print("z",z)
	z1 = Variable(torch.FloatTensor(1,nc_dim).normal_(), requires_grad=True)
	print("z1",z1)
	make_image(model, z,"generated-image")
	make_image(model, z1, "generated-image1")

def make_image(model,z,name):
	x = model.decode(Variable(z.data.cuda(), requires_grad = True))
	x = x.view(1,3,64,64)
	img = x.data.cpu().numpy()
	x1 = img[0,0,:,:]
	x2 = img[0,1,:,:]
	x3 = img[0,2,:,:]
	img_final = np.zeros([64,64,3])
	img_final[:,:,0] = x1
	img_final[:,:,1] = x2
	img_final[:,:,2] = x3
	plt.imshow(img_final, interpolation = 'nearest')
	plt.savefig('./' + name + '.jpg')


train(batchsize = batch_size)
save_model(model)
