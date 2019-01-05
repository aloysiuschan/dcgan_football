# dcgan_football

The Jupyter notebook contains code to build a DCGAN (deep convolutional generative adversarial networks) that generates realistic images of footballers' faces. Training data was obtained from https://www.fmscout.com/a-scope-faces-epl-2018-19.html, and consists of 459 face images of EPL footballers from the 2018-2019 season.

I design the DCGAN using the PyTorch library. The generator is an inverted CNN (convolutional neural network) with a 512-256-128-64 architecture of kernels, where each kernel has a size of 4. Batch normalization and ReLU are applied to the convolutional layers. The length of the random noise input vector is 10.

The discriminator is a CNN with a 64-128-256-512 architecture of kernels, where each kernel has a size of 4. Batch normalization and LeakyReLU are applied to the convolutional layers; unlike ReLU, LeakyReLU allows the pass of a small gradient signal for negative values, hence making the gradients from the discriminator flow stronger into the generator (source: https://sthalles.github.io/advanced_gans/). Training images are scaled to 64x64 resolution (original resolution is 180x180) and fed into the discriminator in batches of 64.

After training the DCGAN for 300 epochs, it managed to produce pretty realistic 64x64 images of footballers' faces, although there are distortions around the hair area for some of the images (possibly because some images in the training set are of footballers with voluminous hairstyles).
