git clone https://github.com/csguoh/MambaIR.git
rm MambaIR/basicsr/archs/mambairv2_arch.py
rm MambaIR/basicsr/data/paired_image_dataset.py
ln -s MambaIR/basicsr/ basicsr
conda env create -f environment.yaml

