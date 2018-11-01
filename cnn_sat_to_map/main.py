import torch 
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import glob
import cv2
import matplotlib.pyplot as plt

num_epochs = 500
batch_size = 8
learning_rate = 0.01
nc_dim = 80
train_data_dir = '../processed_dataset/train'
test_data_dir = '../processed_dataset/val'

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
	
		self.convolution_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True)
		self.convolution_1_in = nn.InstanceNorm2d(num_features=32)
		self.RELU_1 = nn.ReLU(inplace=True)

		self.convolution_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True)
		self.convolution_2_in = nn.InstanceNorm2d(num_features=32)
		self.RELU_2 = nn.ReLU(inplace=True)

		self.convolution_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=1, bias=True)
		self.convolution_3_in = nn.InstanceNorm2d(num_features=64)
		self.RELU_3 = nn.ReLU(inplace=True)

		self.convolution_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=1, bias=True)
		self.convolution_4_in = nn.InstanceNorm2d(num_features=128)
		self.RELU_4 = nn.ReLU(inplace=True)

		self.convolution_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)
		self.convolution_5_in = nn.InstanceNorm2d(num_features=256)
		self.RELU_5 = nn.ReLU(inplace=True)

		self.common_factor_mu = nn.Linear(in_features=1024, out_features=nc_dim, bias=True)
		self.common_factor_dec = nn.Linear(in_features=nc_dim, out_features=2048, bias=True)

		self.relu = nn.ReLU(inplace=True)

		self.deconvolution_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=0, bias=True)
		self.deconvolution_1_in = nn.InstanceNorm2d(num_features=256)
		self.RELU_1_ = nn.ReLU(inplace=True)

		self.deconvolution_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=0, bias=True)
		self.deconvolution_2_in = nn.InstanceNorm2d(num_features=128)
		self.RELU_2_ = nn.ReLU(inplace=True)

		self.deconvolution_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, bias=True)
		self.deconvolution_3_in = nn.InstanceNorm2d(num_features=64)
		self.RELU_3_ = nn.ReLU(inplace=True)

		self.deconvolution_4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, bias=True)
		self.deconvolution_4_in = nn.InstanceNorm2d(num_features=32)
		self.RELU_4_ = nn.ReLU(inplace=True)

		self.deconvolution_5 = nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=4, stride=2, padding=1, bias=True)
		self.sigmoid_final = nn.Sigmoid()

	def encode(self, x):
		x = self.RELU_1(self.convolution_1_in(self.convolution_1(x)))
		x = self.RELU_2(self.convolution_2_in(self.convolution_2(x)))
		x = self.RELU_3(self.convolution_3_in(self.convolution_3(x)))
		x = self.RELU_4(self.convolution_4_in(self.convolution_4(x)))
		x = self.RELU_5(self.convolution_5_in(self.convolution_5(x))) 
		x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
		x = self.common_factor_mu(x)
		return x

	def decode(self, z):
		x = self.relu(self.common_factor_dec(z))
		x = x.view(x.size(0), 512, 2, 2)
		x = self.RELU_1_(self.deconvolution_1_in(self.deconvolution_1(x)))
		x = self.RELU_2_(self.deconvolution_2_in(self.deconvolution_2(x)))
		x = self.RELU_3_(self.deconvolution_3_in(self.deconvolution_3(x)))
		x = self.RELU_4_(self.deconvolution_4_in(self.deconvolution_4(x)))
		x = self.sigmoid_final(self.deconvolution_5(x))
		return x

	def forward(self, x):
		encoded = self.encode(x)
		decoded = self.decode(encoded)
		return decoded, encoded

model = ConvNet().cuda()
l2_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_epoch = 100

def to_img(x):
	x = x.clamp(0, 1)
	x = x.view(batch_size, 3, 128, 128)
	return x

def make_image(x,name):
	x = x.view(128, 128, 3)
	img = x.cpu().numpy()
	plt.imshow(img, interpolation='nearest')
	plt.savefig('./' + name + '.jpg')

def train():
	# load data set and create data loader instance
	print('Loading Sat2Map Dataset...')
	data_dir = '../processed_dataset/train/images/'
	label_dir = '../processed_dataset/train/maps1/'

	dset = datasets.ImageFolder(data_dir, transform=transforms.ToTensor())
	train_data_dir = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=False)

	dset1 = datasets.ImageFolder(label_dir, transform=transforms.ToTensor())
	train_label_dir = torch.utils.data.DataLoader(dset1, batch_size=batch_size, shuffle=False)        

	train_images = []
	train_ground_truths = []
	for i, (img, _) in enumerate(train_data_dir):
		train_images.append(img)
	for i, (img, _) in enumerate(train_label_dir):
		train_ground_truths.append(img)

	len_train = len(train_images)

	for epoch in range(num_epochs):
		model.train().cuda()
		train_loss = 0
		for i in range(len(train_images)):
			img = Variable(train_images[i].cuda())
			map_img = Variable(train_ground_truths[i].cuda()) 
			optimizer.zero_grad()
			reconstructed, encoded = model(img)
			loss = l2_loss(reconstructed, map_img)
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
			save_image(batch_img, './cnn_images/cnn_img/batch_images_{}.png'.format(epoch), nrow=8)
	return model

def test():
	pass

def load_model():
	model.load_state_dict(torch.load('./cnn.pth'))
	return model

def save_model(model):
	torch.save(model.state_dict(), './cnn.pth')


train()
save_model(model)