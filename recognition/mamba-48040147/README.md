# MambaIRv2 Based Greyscale Image Colouriser

![Example Colourisation](./assets/val_example.png)

> Image: An example of the final model's colourisation on one of the validation images. Left: Greyscale input. Right: Colourised output.

## Model and Problem Description

Mamba \[1\] is a state-spaced based architecture designed as an efficient alternative to Transformers for modelling long sequential data such as text.
Unlike Transformers, which use attention mechanisms with quadratic complexity, the original Mamba architecture employs selective state-space models (SSMs) that capture long-range dependencies in linear time, making it faster and more memory-efficient for long sequences.

Here, we focus on MambaIRv2 \[2\], a variation introduced by Guo et al. for image restoration tasks.
The original Mamba processes data sequentially in a causal manner, limiting global information flow across an image.
Inspired by Vision Transformers' non-causal processing enabled by the attention mechanism,
MambaIRv2 introduces an Attentive State-Space Equation (ASE) to enable non-causal, attention-like interactions and a Semantic Guided Neighboring (SGN) module to reorder image tokens by semantic similarity.
These modifications allow single-pass scanning with stronger global context modeling and significantly improved performance and efficiency in visual restoration tasks.

This project takes a pretrained MambaIRv2 model, trained for image super-resolution, and fine tunes it for the task of greyscale image colourisation.
This means the model must produce a "reasonably" coloured image given only the greyscale version of the image as input.

In theory, colourisation is an image restoration task like super-resolution, so the pretrained model should not need heavy modification to perform well on this task.
The model should have already learned to extract relevant image features and use global context to inform pixel-level predictions;
it should just need to learn to take greyscale inputs and map them to colour outputs using the extracted features.

## Model Architecture

![Model Architecture](assets/model_diagram.png)

> Image: The overall model architecture. Modified from \[2\].

As can be seen from the above image, first, a convolution layer is used to extract shallow features.
These features are then passed through a series of Attentive State Space Groups (ASSGs), which are made up
of multiple Attentive State Space Blocks (ASSBs). After the ASSGs, a convolution layer is used to reconstruct the colours.

### Unchanged

Since this project is a fine-tuning task, most of the model architecture was left unchanged.

The MambaIRv2 repository provides a variety of pretrained models of different sizes and for different tasks.
These are all based on the same core architecture, but they differ in terms of number of layers, final layer design and other hyperparameters.
For this project, we used the small SR x2 model as the base model. For elaboration on this choice, see the [Pretrained Model](#pretrained-model) section.

#### Attentive State Space Blocks

These are the fundamental building blocks of the MambaIRv2 architecture.
They are designed to progressively model image dependencies from local to global scales.
Each block has a local and global part, with the local part feeding into the global part.
Each part follows a Norm -> Token Mixer -> Norm -> MLP structure, wrapped by a residual connection with learnable scaling factors to stabilize training and control feature flow.

The token mixer is different for each part:

- The local part uses windowed multi-head self-attention (MHSA) to capture fine-grained spatial relationships within small, non-overlapping regions.
- The global branch employs the Attentive State Space Module (ASSM), which extends Mamba’s state-space formulation to allow global, non-causal information exchange across the entire image sequence. (a) in the image above illustrates the general idea of ASSM and (b) and (c) show a bit more detail about parts of it.

By combining local MHSA and global ASSM within the same block, each ASSB effectively builds hierarchical representations that integrate detailed texture information with broader contextual cues. Both local and global information is cruical for high quality colourisation.

### Changed

The red/pink parts of the above image, i.e. the first and last convolutional layers, are the parts that were modified from the MambaIRv2 small SR x2 model.

Before getting into the details of these layers, we need to mention the colour space used.
There are a variety of colour spaces that can be used to represent images, but the one I chose is the CIELAB colour space.
I will explain why I made this choice in the [Colour Space](#colour-space) section, but for now you just need to know that this colour space has 3 channels:
L (lightness), a (green-red) and b (blue-yellow). Every pixel in a colour image can be represented by these 3 values.

For the input convolution layer, since the input is a greyscale image, it only has 1 channel (lightness).
Thus, I had to change the input channels of the first convolution layer from 3 to 1.
For the initial weights of this layer, I did not just randomly initialise them.
I made them the average of the weights across the 3 input channels of the original model.
I did this because the original model used RGB for the colour space, and each channel contains some information about the lightness of the pixel
so I thought it would be a better initialisation than random weights.

The SR x2 model's last layer has a few convolutions and an upsampler. Since this project does not increase the image resolution, this had to be changed.
Also, since the input is already the lightness channel, the output only needs to predict the a and b channels.
Thus, I made the last layer a simple 3x3 convolution that outputs 2 channels (a and b).
I let these weights just be randomly initialised since there is no obvious way to use the pretrained weights here.

## Dependencies and Reproducibility

Conda is recommended for managing dependencies because there is an [environment.yml](./environment.yaml) file that has all of the required packages and their versions that you can use to easily set up the environment.

Assuming your environment has `git`, `conda`, `ln`, and `wget`, you can just run the [setup.sh](./setup.sh) script to download all dependencies and set up the environment. This clones the MambaIRv2 repository, creates the conda environment, and downloads the dataset and pretrained weights.

To set up the training, validation and test data:

1. Unzip `DIV2K_train_LR_bicubic_X2.zip` and `DIV2K_valid_LR_bicubic_X2.zip`.
2. Create an `images` folder and inside that make `train`, `val` and `test` folders.
3. Move the images specified in [Dataset and Pre-Processing](#dataset-and-pre-processing) into the respective folders.
4. Download the [test images](https://github.com/gayanku/greyscale-colorization) and move them into the `test` folder.

To activate the conda environment, run

```bash
conda activate mambair
```

To train the model, run

```bash
python train.py
```

To run inference using the trained model on `image_name.png`, first copy the model weights to `./mambairv2_Colouriser_Final.pth` and then run

```bash
python predict.py image_name.png
```

The output image will be saved as `images/output/image_name.png`.

Running just

```bash
python predict.py
```

will run inference on all images in the `test` folder.

A set seed has been set up so the results should be reproducible. All training and testing was done on Rangpur using a single A100 GPU with 40GB VRAM. Training would have been faster if multiple GPUs were used but I did not want to hog too many resources.

## Dataset and Pre-Processing

The dataset I chose to use is [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) \[3\] because it was used to train the MambaIRv2 SR models and most denoising models.
I did not use a larger dataset (the MambaIRv2 SR for example are also trained on Flickr2K) due to time and resource constraints (Rangpur has limited storage)
and since this is a fine tuning task, I thought a smaller dataset would suffice.
DIV2K is composed of a number of high quality colour photographs so it makes sense for this task.

However, some of the images in the DIV2K dataset are greyscale or at least lack a lot of colour. These would be a detriment to fine tuning a colourisation model so
I removed them. The images I removed are:

- 0005x2.png
- 0014x2.png
- 0043x2.png
- 0138x2.png
- 0188x2.png
- 0190x2.png
- 0202x2.png
- 0230x2.png
- 0769x2.png
- 0800x2.png
- 0843x2.png
- 0845x2.png
- 0846x2.png
- 0868x2.png
- 0892x2.png

I chose to use the 2X downscaled images instead of the original 2K resolution images because

1. The model I am fine tuning was originally trained on these downscaled images as input so it would be "used" to them
2. The 2K images take a lot of storage which is limited on Rangpur
3. While training, 128x128 crops are used so with a smaller original image, each crop contains a larger portion of the image which may help with learning to use more global context
4. For validation, the full size images are too large to fit in memory

DIV2K comes already split into training and validation sets but I decided to change the split.
I kept the first 22 validation images (0801x2.png - 0822x2.png) as the validation set and moved the rest to the training set.
I did this so that the training set had more data and because validation uses the whole image instead of just a crop so you can view
it as not needing as many images. I also wanted a smaller validation set so that validation would be faster.

This meant the training set had 863 images and the validation set had 22 images.

During training, 100 random 128x128 crops are taken from each image and they are randomly flipped and/or rotated. This was done in training all of the original MambaIRv2 models as well. Cropping was necessary due to memory constraints. If the images were instead resized, a lot of texture would be lost which is important for predicting colour. The multiple random crops per image and augmentations effectively increases the amount of training data and helps the model generalise better. Furthermore, all random aspects change each epoch which further increases the effective dataset size.

Each training epoch thus has 86300 samples.

Finally, each image is converted from RGB to CIELAB. The L channel is scaled to \[0, 1\] by dividing by 100, and the a and b channels are scaled to approximately \[-1, 1\] by dividing by 127. Neural networks tend to perform better when inputs and outputs are scaled to smaller ranges since weights can be kept smaller so gradients do not explode as easily.

## Training Procedure

Many of the MambaIRv2 were actually not trained from scratch but rather fine tuned from other pretrained MambaIRv2 models.
For example, the small SR x3 model was fine tuned from the small SR x2 model.
Hence, I thought it was reasonable to fine tune for this project with a similar method.

MambaIRv2 mainly focuses on number of iterations rather than number of epochs so I did the same. I used a batch size of 3 since that was the largest that could fit in memory. This means each epoch is 28767 iterations. It may have been better to use gradient accumulation to simulate a larger batch size, but the original MambaIRv2 models used a batch size of 4 so I thought 3 would be acceptable.

I trained the model for 200000 total iterations. I originally tried 250000 since that was what the small SR x3 model used but it looked like this would have taken more than 2 and a half days so I reduced it. In total, there were then 7 epochs (although the 7th epoch stopped early since it stopped at 200000 iterations).

When fine tuning begins, initially all parameters of the model are frozen except for the first and last convolution layers. MambaIRv2 did not do this but it seemed to be a good idea because the first and last layers would have to change a lot, whereas the other layers would not have to change much since they were already trained for image restoration. Freezing the other layers makes training faster and reduces the chance of the model forgetting what it had already learned. After 60000 iterations, with a now lower learning rate (see below), all parameters are then unfrozen to allow the whole model to be fine tuned for colourisation.

I used the Adam optimiser with an inital learning rate of 0.0002 and used a scheduler to make this halve at iterations 50000, 100000, 160000, 180000 and 190000. I chose these values because they are scaled versions of what the small SR x3 model used, except that I added the 50000 milestone at the start and made the initial learning rate 0.0002 instead of 0.0001 so that the model could do the inital learning of the first and last layers faster (since they were not pretrained). I also used a weight decay of 0 and beta values of 0.9 and 0.99 as per the original MambaIRv2 model's training.

### Loss Function

The pretrained model I was using at first (see [below](#pretrained-model)) used Charbonnier loss with $\epsilon = 0.001$ so that is what I used.
When I switched to using the small SR x2, I kept using Charbonnier loss because I saw no reason to change it.
Charbonnier loss is usually defined as $\sqrt{x^2 + \epsilon^2}$ where $x$ is the difference between the predicted and target values \[4\].
It acts like a smooth approximation to L1 loss and is differentiable at 0, which helps with stable training.

At this point, the colourisations produced by the trained model were very dull and brown or grey a lot of the time.
I noticed that the loss values I got never went below 0.03 and so I investigated the implementation of Charbonnier loss I was using (from the MambaIRv2 repository).
It defined Charbonnier loss as $\sqrt{x^2 + \epsilon}$, not squaring the epsilon.
This means that it was mathematically impossible for the loss to go below 0.0316 ($\sqrt{0.001} \approx 0.0316$) and since the gradient is very small around this point,
the model could not really improve further.

Instead of just reducing epsilon, I decided to switch to L1 loss ($|x|$) since it is what the SR models used
and it can lead to faster convergence since the gradient does not approach 0 (except at 0).

This alone would probably not have significantly improved the colours produced however.
It is known that regression losses (like L1 or Charbonnier) tend to produce desaturated colours because
they encourage conservative estimates that minimise average error \[5\]\[6\].
To achieve more vibrant colours, using another loss would be beneficial.

I think that introducing a discriminative loss, essentially changing the model into a conditional GAN, could be great for this task \[5\].
If the colouriser always produces bland images, the discriminator would learn to easily identify them as fake and so the loss would be high.
Thus, this would encourage the colouriser to produce more vibrant colours.
I did not implement this however due to time constraints.

Another approach that seemed promising to me was what Zhang et al. did \[6\].
Instead of predicting the a and b values directly, they predicted a distribution over quantized ab values (313 bins).
Then, they used a cross-entropy loss to train the model to predict the correct bin.
This encourages the model to consider multiple plausible colours for each pixel rather than averaging them out.
There are many situations where multiple colours are plausible for a given greyscale value (e.g. an apple could be red or green)
so it would be better to model this rather than just predicting the average colour.
They also used a weight in the loss to emphasise rare colours (the loss is based on the inverse frequency of colours in the training data)
so that the model does not just learn to predict common colours like green, blue or grey all the time.

I did not implement this either, again due to time, but it inspired me to think about colours as categories
which led to the following idea:
$$L_{WCC}((a_t, b_t), (a_p, b_p)) = \sigma \left(k \left(\sqrt{(a_t - a_p)^2 + (b_t - b_p)^2} - \delta\right)\right)$$
where $(a_t, b_t)$ are the true ab values, $(a_p, b_p)$ are the predicted ab values, $\sigma(x) = \frac{1}{1+e^{-x}}$ is the sigmoid function and $k$ and $\delta$ are constants.

$L_{WCC}$ is what I call "wrong colour category" loss. I originally thought to just use a step function: 0 loss when the difference is within $\delta$ (the idea is everything in $\delta$ radius is the same colour category) and 1 loss otherwise. This loss treats any colour that is the wrong category as the same so it penalises always just choosing a conservative guess. E.g. if the true colour could sometimes be red and sometimes green, if the model predicts brown all of the time it will always have a high loss but if it predicts say red then the loss will be lower. I believe this should lead to more vibrant colours. $L_{WCC}$ does not use a step function however because that is not differentiable so I used a sigmoid to approximate it. I used euclidean distance in ab space since in CIELAB, euclidean distance corresponds to perceptual difference.

This loss function would not be reasonable on its own since the gradient for incorrect predictions is very small so the model would struggle to learn.

The overall loss function I used is thus:
$$L_1 + \lambda L_{WCC}$$

$L_1$ is the mean L1 loss over every pixel's a and b values and $L_{WCC}$ is the mean wrong colour category loss over every pixel.
In theory, $L_1$ encourages overall accuracy while $L_{WCC}$ encourages more vibrant colours.

The specific hyperparameters I used were $k=30$, $\delta=0.08$ and $\lambda=0.2$.
Due to a lack of time, these are the only combinations of values I tried.
To be honest, I got these values from ChatGPT after explaining my idea to it.
I looked at the graph on [Desmos](https://www.desmos.com/calculator/mmnb14hywi) to try to see if these looked reasonable
and I thought it did, though this graph is not really accurate since it just looks at one channel of a single pixel's difference.
One rough way to see that $\delta=0.08$ is probably reasonable is that \[6\] used 313 ab bins. Our normalised ab space goes from roughly -1 to 1
so if we assume the space is a square (which it isn't and also it is dependent on the L value) then each bin would be about 0.11 units wide, which
is close to 0.08.

## Results

### Training and Validation Loss

| Total Loss                                                          | L1 Loss                                                       | $L_{WCC}$ Loss                                                  |
| ------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------- |
| ![Training Total Loss](./assets/loss/train_total.png)               | ![Training L1 Loss](./assets/loss/train_l1.png)               | ![Training LWCC Loss](./assets/loss/train_wc.png)               |
| ![Smooth Training Total Loss](./assets/loss/train_total_smooth.png) | ![Smooth Training L1 Loss](./assets/loss/train_l1_smooth.png) | ![Smooth Training LWCC Loss](./assets/loss/train_wc_smooth.png) |
| ![Validation Total Loss](./assets/loss/val_total.png)               | ![Validation L1 Loss](./assets/loss/val_l1.png)               | ![Validation LWCC Loss](./assets/loss/val_wc.png)               |

### Test Images

| Input Greyscale                                                                                                 | Output Colourised                      |
| --------------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| ![G_1](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_1.jpg)   | ![G_1](./assets/test-output/G_1.jpg)   |
| ![G_2](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_2.jpg)   | ![G_2](./assets/test-output/G_2.jpg)   |
| ![G_3](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_3.jpg)   | ![G_3](./assets/test-output/G_3.jpg)   |
| ![G_4](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_4.jpg)   | ![G_4](./assets/test-output/G_4.jpg)   |
| ![G_5](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_5.jpg)   | ![G_5](./assets/test-output/G_5.jpg)   |
| ![G_6](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_6.jpg)   | ![G_6](./assets/test-output/G_6.jpg)   |
| ![G_7](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_7.jpg)   | ![G_7](./assets/test-output/G_7.jpg)   |
| ![G_8](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_8.jpg)   | ![G_8](./assets/test-output/G_8.jpg)   |
| ![G_9](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_9.jpg)   | ![G_9](./assets/test-output/G_9.jpg)   |
| ![G_10](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_10.jpg) | ![G_10](./assets/test-output/G_10.jpg) |
| ![G_11](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_11.jpg) | ![G_11](./assets/test-output/G_11.jpg) |
| ![G_12](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_12.jpg) | ![G_12](./assets/test-output/G_12.jpg) |
| ![G_13](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_13.jpg) | ![G_13](./assets/test-output/G_13.jpg) |
| ![G_14](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_14.jpg) | ![G_14](./assets/test-output/G_14.jpg) |
| ![G_15](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_15.jpg) | ![G_15](./assets/test-output/G_15.jpg) |
| ![G_16](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_16.jpg) | ![G_16](./assets/test-output/G_16.jpg) |
| ![G_17](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_17.jpg) | ![G_17](./assets/test-output/G_17.jpg) |
| ![G_18](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_18.jpg) | ![G_18](./assets/test-output/G_18.jpg) |
| ![G_19](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_19.jpg) | ![G_19](./assets/test-output/G_19.jpg) |
| ![G_20](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_20.jpg) | ![G_20](./assets/test-output/G_20.jpg) |
| ![G_21](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_21.jpg) | ![G_21](./assets/test-output/G_21.jpg) |
| ![G_22](https://raw.githubusercontent.com/gayanku/greyscale-colorization/refs/heads/main/test-dataset/G_22.jpg) | ![G_22](./assets/test-output/G_22.jpg) |

### Observations

| 100k Iterations                      | 150k Iterations                      | 200k Iterations                      |
| ------------------------------------ | ------------------------------------ | ------------------------------------ |
| ![100k](./assets/reddening/100k.png) | ![150k](./assets/reddening/150k.png) | ![200k](./assets/reddening/200k.png) |

| Earlier Model (no $L_{WCC}$)     | 100k Iterations                | 150k Iterations                       |
| -------------------------------- | ------------------------------ | ------------------------------------- |
| ![earlier](./assets/G_1_old.png) | ![100k](./assets/G_1_100k.jpg) | ![150k](./assets/test-output/G_1.jpg) |

## Design Decisions

### Pretrained Model

The MambaIRv2 model that I originally used as the base model was the Colour DN 15.
This model is designed for denoising colour images with a noise level of 15.

I chose it because unlike super-resolution, denoising does not change the image dimensions, like with colourisation.

This model was quite slow to fine tune. I think a large part of this was that due to the size of the model, I could only use a batch size of 1.

I thus decided to switch to a smaller model. The only smaller models were the light and small SR models.
The light model has significantly fewer parameters so I thought it may not perform as well so I chose the small SR x2 model.
The reason I chose the x2 model over the x3 or x4 models is that its what the x3 and x4 models were fine tuned from and since
it was less trained overall, it may be less specialised to super-resolution and thus better for fine tuning to colourisation.

### Colour Space

I knew from the start that I wanted to use a colour space that separated lightness from colour information.
This makes the model's task easier since it can just take in the lightness channel and predict the colour channels.

I initially chose YCbCr because of its simplicity. When reading an image as greyscale, the values are already the Y channel.
Each channel is also just a linear combination of RGB so converting between RGB and YCbCr is easy.

However, I later switched to CIELAB, primarily from seeing its use in \[6\].
Although CIELAB-RGB conversion is mathematically more complex, from a coding perspective it is not when using a library.
Furthermore, CIELAB is designed to be perceptually uniform, meaning that euclidean distances in this space correspond to perceptual differences.
This is not true for YCbCr.
Using CIELAB should then result in better colour predictions since the loss function will better reflect perceptual colour differences.

## References

- \[1\] Gu, A., & Dao, T. (2024, May 31). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. ArXiv. https://doi.org/10.48550/arXiv.2312.00752
- \[2\] Guo, H., Guo, Y., Zha, Y., Zhang, Y., Li, W., Dai, T., Xia, S.-T., & Li, Y. (2025, March 11). MambaIRv2: Attentive State Space Restoration. ArXiv. https://arxiv.org/abs/2411.15269
- \[3\] Timofte, R., Augustsson, E., Gu, S., Wu, J., Ignatov, A., & Gool, L. V. (2017). DIV2K Dataset. Data.vision.ee.ethz.ch. https://data.vision.ee.ethz.ch/cvl/DIV2K/
- \[4\] Lee, V. (2020, July 2). Loss Functions. Vivian’s Machine Learning Note’s. https://machine-learning-note.readthedocs.io/en/latest/basic/loss_functions.html
- \[5\] Goree, S. (2021, April 21). The Limits of AI Image Colorization: A Companion. Sam Goree. https://samgoree.github.io/2021/04/21/colorization_companion.html
- \[6\] Zhang, R., Isola, P., & Efros, A. A. (2016, October 5). Colorful Image Colorization. ArXiv. https://arxiv.org/abs/1603.08511
