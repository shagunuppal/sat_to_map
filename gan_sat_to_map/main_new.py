import os
import argparse
import numpy as np
from scipy.misc import imread
from itertools import cycle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable


import gc
from utils import weights_init
import matplotlib.pyplot as plt
from utils import transform_config
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from utils import imshow_grid, mse_loss, reparameterize, l1_loss
from network import Generator, Discriminator
from torchvision.utils import save_image

batch_size = 8
num_epochs = 500
image_size = 128

generator = Generator(nc_dim = 80)
generator.apply(weights_init)
generator = generator.cuda()

discriminator = Discriminator()
discriminator.apply(weights_init)
discriminator = discriminator.cuda()

ones_label = torch.ones(batch_size)
ones_label = Variable(ones_label.cuda())
zeros_label = torch.zeros(batch_size)
zeros_label = Variable(zeros_label.cuda())

loss = nn.BCEWithLogitsLoss()

def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 128, 128)
    return x

D_solver = optim.Adam(discriminator.parameters(), lr=0.005, betas=(0.5, 0.999))
G_solver = optim.Adam(generator.parameters(), lr=0.005, betas=(0.5, 0.999))

def train():
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

    img = torch.FloatTensor(batch_size, 3, image_size, image_size)
    map_img = torch.FloatTensor(batch_size, 3, image_size, image_size)

    train_images = []
    train_ground_truths = []
    for i, (img1, _) in enumerate(train_data_dir):
        train_images.append(img1)
    for i, (img1, _) in enumerate(train_label_dir):
        train_ground_truths.append(img1)

    for i, (img1, _) in enumerate(train_data_dir_more):
        train_images.append(img1)
    for i, (img1, _) in enumerate(train_label_dir_more):
        train_ground_truths.append(img1)
    
    train_ground_truths = train_ground_truths[:len(train_ground_truths)-2]
    train_images = train_images[:len(train_images)-2]

    len_train = len(train_images)

    for epoch in range(num_epochs):
        for i in range(len_train):
            #print(type(train_images[i]), type(img))

            img.copy_(train_images[i])
            map_img.copy_(train_ground_truths[i])

            img = img.cuda()
            img_1 = Variable(img.cuda())
            map_img = map_img.cuda()
            map_img_1 = Variable(map_img.cuda())


            #img = Variable(train_images[i].cuda())
            #map_img = Variable(train_ground_truths[i].cuda())

            # discriminator
            discriminator.zero_grad()
            # real
            D_real = discriminator(img_1, map_img_1)
            ones_label.data.resize_(D_real.size()).fill_(1)
            zeros_label.data.resize_(D_real.size()).fill_(0)
            D_loss_real = loss(D_real, ones_label)
            D_x_y = D_real.data.mean()

            # fake
            G_fake = generator(img_1)
            D_fake = discriminator(img_1, G_fake.detach())
            D_loss_fake = loss(D_fake, zeros_label) #loss(D_real, ones_label)
            D_x_gx = D_fake.data.mean()

            D_loss = D_loss_real + D_loss_fake
            D_loss.backward()
            D_solver.step()

            # generator
            generator.zero_grad()
            G_fake = generator(img_1)
            D_fake = discriminator(img_1, G_fake)
            D_x_gx_2 = D_fake.data.mean()
            G_loss = loss(D_fake, ones_label) + 100 * F.smooth_l1_loss(G_fake, map_img_1)
            G_loss.backward()
            G_solver.step()


            if(i%100==0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tD Loss: {:.6f}\tG Loss: {:.6f}'.format(
                    epoch,
                    i * len(img_1),
                    len_train, 
                    100. * i / len_train,
                    D_loss.data[0] / len(img_1),
                    G_loss.data[0] / len(img_1)
                    ))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, (D_loss+G_loss).data[0] / len_train))
        if epoch % 10 == 0:
            ground_truth_img = to_img(img_1.cpu().data)
            ground_truth_map = to_img(map_img_1.cpu().data)
            reconstructed = generator(img_1)
            rec = to_img(reconstructed.cpu().data)
            batch_img = torch.zeros(24, 3, 128, 128)
            batch_img[0:8, :, :, :] = ground_truth_img
            batch_img[8:16, :, :, :] = ground_truth_map
            batch_img[16:24, :, :, :] = rec
            save_image(batch_img, './gan_img/batch_images_{}.png'.format(epoch), nrow=8)
    return generator, discriminator

gen, disc = train()
