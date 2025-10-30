git clone https://github.com/csguoh/MambaIR.git
ln -s MambaIR/basicsr/ basicsr
conda env create -f environment.yaml
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X2.zip
wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X2.zip
wget https://github.com/csguoh/MambaIR/releases/download/v1.0/mambairv2_classicSR_Small_x2.pth

