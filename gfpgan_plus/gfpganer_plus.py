import os
import numpy as np
import cv2
from PIL import Image
import torch
from .model.generator import GFPGANv1Clean


class GFPGANerPlus:
    """Class for running GFPGANv1Clean model."""

    def __init__(self, model_path, device=None):
        """Initialize the model and load weights."""
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.netG = GFPGANv1Clean(
            out_size=512,
            channel_multiplier=2,
            fix_decoder=False,
            input_is_latent=True,
            different_w=True,
            sft_half=True
        ).to(self.device)
        state_dict = torch.load(model_path)['g_ema']
        self.netG.load_state_dict(state_dict)
        self.netG.eval()
        self.input_shape = (512, 512)

    def run(self, img_path):
        """Run the model on the input image and return the output."""
        inp = self.preprocess(img_path)
        with torch.no_grad():
            oup, _ = self.netG(inp)
        oup = self.postprocess(oup)
        return oup

    def preprocess(self, img):
        """Preprocess the image before feeding it to the model.

        Args:
            img: The input image, which can be a string path, a NumPy array, or a PIL image.

        Returns:
            A torch.Tensor representing the preprocessed image.
        """
        is_pil = False
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, Image.Image):
            img = np.array(img)
            is_pil = True
        elif not isinstance(img, np.ndarray):
            raise ValueError("Unsupported image type. Provide a string path, a NumPy array, or a PIL image.")

        if img.shape[-1] == 3 and not is_pil:  # Check if the image has 3 channels (RGB)
            img = img[..., ::-1]  # Convert from RGB to BGR if necessary

        if img.shape[:2] != self.input_shape:
            img = cv2.resize(img, (512, 512))
        img = img.astype(np.float32) / 127.5 - 1
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis, ...])
        return img_tensor.to(self.device)

    def postprocess(self, img):
        """Postprocess the model output image."""
        img_clipped = torch.clip(img, -1, 1)[0]
        img_np = img_clipped.permute(1, 2, 0).cpu().numpy()[..., ::-1]
        return (img_np + 1) * 127.5
