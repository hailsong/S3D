# sketch2face3D

```
git clone --recurse-submodules https://github.com/hailsong/sketch2face3D.git
```
It will recursively clone the repository of pix2pix3D, which is our baseline.
We further update its key features by inheriting and re-packaging.

### Goal
Sketch to 3D Modeling (of face dataset)

### TODO List
- [x] pix2pix3D inference demo - Hail
- [ ] pix2pix3D training code check (with CelebA mask data) - Hail
- [x] Making Edge dataset for CelebA - Soomin
- [ ] pix2pix3D training code check (with CelebA edge data) - Hail
- [ ] Building U-net based Edge2Mask NN
- [ ] Building U-net based Edge2Mask NN (By sharing style vector of pix2pix3D Encoder)
- [ ] Using 3D Gaussian Splatting for Neural Rendering part
- [ ] Using Diffusion based methods for the original GAN part
