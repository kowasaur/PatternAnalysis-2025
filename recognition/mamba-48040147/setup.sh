git clone https://github.com/csguoh/MambaIR.git
ln -s MambaIR/basicsr/ basicsr
# TODO: change environment.yaml to install packaging through conda, not pip
conda env create -f MambaIR/environment.yaml

