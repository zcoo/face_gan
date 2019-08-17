import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import visdom

import torch.nn as nn
import torch.nn.functional as F
import torch
from dataset import Dataset
import GAN
import utils

os.makedirs("images", exist_ok=True)

opt = utils.get_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = GAN.Generator(opt.latent_dim, img_shape)
discriminator = GAN.Discriminator(img_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)

data_dirs = ["./MTFL/AFLW", "./MTFL/lfw_5590", "./MTFL/net_7876"]
dataset = Dataset(data_dirs)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if opt.if_visual:
    vis = visdom.Visdom(env='mygan')

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        x = torch.Tensor([epoch+0.003*i])
        y_d_loss = torch.Tensor([d_loss.item()])
        y_g_loss = torch.Tensor([g_loss.item()])
        y_sum_loss = torch.Tensor([d_loss.item() + g_loss.item()])
        vis.line(X=x, Y=y_d_loss, win="dloss", update='append', opts={'title': 'y=d_loss'})
        vis.line(X=x, Y=y_g_loss, win="gloss", update='append', opts={'title': 'y=g_loss'})
        vis.line(X=x, Y=y_sum_loss, win="d_g_loss", update='append', opts={'title': 'y=sum_loss'})

        if epoch % opt.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer': optimizer_G.state_dict(),
                'discriminator_optimizer': optimizer_D.state_dict(),
            }, "save/face_model_%d.pkl" % epoch)

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
