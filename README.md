# GFPGAN-Plus

This package provides tools for image restoration using GANs, including implementations of StyleGAN2 and GFPGAN.

## Installation

```bash
pip install git+https://github.com/datvtn/GFPGAN-Plus.git
```

## USage

```bash
from gfpgan_plus import GFPGANerPlus

model = GFPGANerPlus("../gfpgan_1024.pth")
output_image = model.run('../Adele_crop.png')
```
