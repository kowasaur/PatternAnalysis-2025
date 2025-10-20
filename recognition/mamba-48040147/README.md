Talk about having to decide which pretrained model to use as a base. Advantages and disadvantages
Talk about stuff that makes using a pretrained model hard

- both sr and denoising really only need local features whereas colour needs more context
- Adding on to the above, sr and denoising don't really need to have a massive amount of training data, but with colour it would because if it has never seen something it probably can't guess it's colour. For example a tiger. If a tiger is not in the training data, how is it supposed to know it's orange with black stripes? Obviously some of it is contained in the lightness (like it can probably tell it's black stripes) but the orange part is tricky. Maybe if it had seen a lion or something it could guess but like what if the training data had no big cats?

Talk about dataset. Turns out some of the training data is black and white!

# Sources:

- https://github.com/csguoh/MambaIR/
- https://data.vision.ee.ethz.ch/cvl/DIV2K/
