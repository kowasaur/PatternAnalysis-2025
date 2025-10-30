# MambaIRv2 Based Greyscale Image Colouriser

Image of greyscale and colour image side by side

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

Talk about dataset. Turns out some of the training data is black and white!

## Training Procedure

## Results

Training plot and sample test results

## Design Decisions

### Pretrained Model

Talk about having to decide which pretrained model to use as a base. Advantages and disadvantages
Talk about stuff that makes using a pretrained model hard

- both sr and denoising really only need local features whereas colour needs more context
- Adding on to the above, sr and denoising don't really need to have a massive amount of training data, but with colour it would because if it has never seen something it probably can't guess it's colour. For example a tiger. If a tiger is not in the training data, how is it supposed to know it's orange with black stripes? Obviously some of it is contained in the lightness (like it can probably tell it's black stripes) but the orange part is tricky. Maybe if it had seen a lion or something it could guess but like what if the training data had no big cats?

### Colour Space

## References

- \[1\] Gu, A., & Dao, T. (2024, May 31). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. ArXiv. https://doi.org/10.48550/arXiv.2312.00752
- \[2\] Guo, H., Guo, Y., Zha, Y., Zhang, Y., Li, W., Dai, T., Xia, S.-T., & Li, Y. (2025, March 11). MambaIRv2: Attentive State Space Restoration. ArXiv. https://arxiv.org/abs/2411.15269

- https://data.vision.ee.ethz.ch/cvl/DIV2K/
- https://arxiv.org/pdf/1603.08511
