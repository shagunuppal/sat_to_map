import os
import argparse
import numpy as np
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable

import gc
from utils import weights_init
import matplotlib.pyplot as plt
from utils import transform_config
from torch.utils.data import DataLoader
from utils import imshow_grid, mse_loss, reparameterize, l1_loss
from network import Encoder, Decoder, Discriminator

latent_dim = 80

encoder = Encoder(latent_dim)
encoder.apply(weights_init)

decoder = Decoder(latent_dim)
decoder.apply(weights_init)

discriminator = Discriminator()
discriminator.apply(weights_init)

criteron = nn.BCELoss()

optimizer_G = optim.Adam(decoder.parameters(), lr = learning_rate_gen)#, beta = beat_gen)
optimizer_D = optim.Adam(discriminator.parameters(), lr = learning_rate_disc)#, beta = beta_disc)

def to_img(x):
	x = x.clamp(0, 1)
	x = x.view(x.size(0), 3, 128, 128)
	return x

def train(batch_size):
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
			#map_img = Variable(train_ground_truths[i].cuda()) 

			
			
			
