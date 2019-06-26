import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import numpy as np
from utils import imshow_grid

from itertools import cycle
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import transform_config, reparameterize


class Encoder(nn.Module):
	def __init__(self, nc_dim):
		super(Encoder, self).__init__()

		self.conv_model = nn.Sequential(OrderedDict([
			('convolution_1',
			 nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True)),
			('convolution_1_in', nn.InstanceNorm2d(num_features=32)),
			('ELU_1', nn.ReLU(inplace=True)),

			('convolution_2',
			 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_2_in', nn.InstanceNorm2d(num_features=32)),
			('ELU_2', nn.ReLU(inplace=True)),

			('convolution_3',
			 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_3_in', nn.InstanceNorm2d(num_features=64)),
			('ELU_3', nn.ReLU(inplace=True)),

			('convolution_4',
			 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_4_in', nn.InstanceNorm2d(num_features=128)),
			('ELU_4', nn.ReLU(inplace=True)),

			('convolution_5',
			 nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)),
			('convolution_5_in', nn.InstanceNorm2d(num_features=256)),
			('ELU_5', nn.ReLU(inplace=True))
		]))

		self.common_factor = nn.Linear(in_features=1024, out_features=nc_dim, bias=True)

	def forward(self, x):
		x = self.conv_model(x)
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

		nv_latent_space_mu = self.varying_factor_mu(x)
		nv_latent_space_logvar = self.varying_factor_logvar(x)

		nc_latent_space = self.common_factor(x)

		return nv_latent_space_mu, nv_latent_space_logvar, nc_latent_space


class Decoder(nn.Module):
	def __init__(self, nc_dim):
		super(Decoder, self).__init__()

		self.common_factor = nn.Linear(in_features=nc_dim, out_features=1024, bias=True)

		self.relu = nn.ReLU(inplace=True)

		self.deconv_model = nn.Sequential(OrderedDict([
			('deconvolution_1',
			 nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)),
			('deconvolution_1_in', nn.InstanceNorm2d(num_features=256)),
			('ELU_1', nn.ReLU(inplace=True)),

			('deconvolution_2',
			 nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=0, bias=True)),
			('deconvolution_2_in', nn.InstanceNorm2d(num_features=128)),
			('ELU_2', nn.ReLU(inplace=True)),

			('deconvolution_3',
			 nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)),
			('deconvolution_3_in', nn.InstanceNorm2d(num_features=64)),
			('ELU_3', nn.ReLU(inplace=True)),

			('deconvolution_4',
			 nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True)),
			('deconvolution_4_in', nn.InstanceNorm2d(num_features=32)),
			('ELU_4', nn.ReLU(inplace=True)),

			('deconvolution_5',
			 nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True)),
			('sigmoid_final', nn.Sigmoid())

		]))

	def forward(self, nc_latent_space):
		nc_latent_space = self.relu(self.common_factor(nc_latent_space))

		x = x.view(x.size(0), 512, 2, 2)
		x = self.deconv_model(x)

		return x

class Generator(nn.Module):
	def __init__(self, nc_dim):
		super(Generator, self).__init__()
		self.conv_model = nn.Sequential(OrderedDict([
			('convolution_1',
			 nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True)),
			('convolution_1_in', nn.InstanceNorm2d(num_features=32)),
			('ELU_1', nn.ReLU(inplace=True)),

			('convolution_2',
			 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_2_in', nn.InstanceNorm2d(num_features=32)),
			('ELU_2', nn.ReLU(inplace=True)),

			('convolution_3',
			 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_3_in', nn.InstanceNorm2d(num_features=64)),
			('ELU_3', nn.ReLU(inplace=True)),

			('convolution_4',
			 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_4_in', nn.InstanceNorm2d(num_features=128)),
			('ELU_4', nn.ReLU(inplace=True)),

			('convolution_5',
			 nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)),
			('convolution_5_in', nn.InstanceNorm2d(num_features=256)),
			('ELU_5', nn.ReLU(inplace=True))
		]))

		self.fc1 = nn.Linear(in_features=1024, out_features=nc_dim, bias=True)

		self.fc2 = nn.Linear(in_features=nc_dim, out_features=1024*2, bias=True)

		self.relu = nn.ReLU(inplace=True)

		self.deconv_model = nn.Sequential(OrderedDict([
			('deconvolution_1',
			 nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)),
			('deconvolution_1_in', nn.InstanceNorm2d(num_features=256)),
			('ELU_1', nn.ReLU(inplace=True)),

			('deconvolution_2',
			 nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=0, bias=True)),
			('deconvolution_2_in', nn.InstanceNorm2d(num_features=128)),
			('ELU_2', nn.ReLU(inplace=True)),

			('deconvolution_3',
			 nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)),
			('deconvolution_3_in', nn.InstanceNorm2d(num_features=64)),
			('ELU_3', nn.ReLU(inplace=True)),

			('deconvolution_4',
			 nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True)),
			('deconvolution_4_in', nn.InstanceNorm2d(num_features=32)),
			('ELU_4', nn.ReLU(inplace=True)),

			('deconvolution_5',
			 nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True)),
			('sigmoid_final', nn.Sigmoid())

		]))

	def forward(self, x):
		x = self.conv_model(x)
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		x = self.fc1(x)
		x = self.relu(self.fc2(x))
		x = x.view(x.size(0), 512, 2, 2)
		x = self.deconv_model(x)
		return x


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		self.conv_model = nn.Sequential(OrderedDict([
			('convolution_1',
			 nn.Conv2d(in_channels=6, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_1_in', nn.InstanceNorm2d(num_features=32)),
			('LeakyReLU_1', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

			('convolution_2',
			 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_2_in', nn.InstanceNorm2d(num_features=64)),
			('LeakyReLU_2', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

			('convolution_3',
			 nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_3_in', nn.InstanceNorm2d(num_features=128)),
			('LeakyReLU_3', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

			('convolution_4',
			 nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)),
			('convolution_4_in', nn.InstanceNorm2d(num_features=128)),
			('LeakyReLU_4', nn.LeakyReLU(negative_slope=0.2, inplace=True)),

			('convolution_5',
			 nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)),
			('convolution_5_in', nn.InstanceNorm2d(num_features=256)),
			('LeakyReLU_5', nn.LeakyReLU(negative_slope=0.2, inplace=True))
		]))

		self.fully_connected_model = nn.Sequential(OrderedDict([
			('output', nn.Linear(in_features=1024, out_features=2, bias=True))
		]))

	def forward(self, image_1, image_2):
		x = torch.cat((image_1, image_2), dim=1)
		x = self.conv_model(x)

		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		x = self.fully_connected_model(x)

		return x
