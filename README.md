# Reproduction of Image Inpainting Using Fractional-Order Non-Linear Diffusion

## Authors

- **Jay Nidumolu (21108165)**
- **Praveen Raavi(21007013)**


## Overview
This repository contains the implementation of an image inpainting model based on fractional-order nonlinear diffusion. The project extends the original methodology, which was designed for 2D grayscale images, to RGB images, addressing the complexities of multi-channel data. The model demonstrates superior performance in restoring missing regions, preserving edges, and minimizing artifacts such as staircase and speckle effects.

---

## Key Features
- **Fractional-Order Diffusion**: Balances edge preservation and artifact suppression.
- **Multi-Channel Support**: Extended to handle RGB images using:
  - Channel Separation Approach
  - Direct 3D Signal Approach
- **Edge-Driven Mask Generation**: Automatically identifies regions for inpainting.
- **Comparative Evaluation**: Benchmarked against the Total Variation (TV) model and Bilateral Telea method.


---

## How to Use
### 1. Clone the Repository
```bash
git clone https://github.com/jay-nidumolu/Image_Inpainting_using_fractional-order_diffusion_method.git
cd Image_Inpainting_using_fractional-order_diffusion_method
