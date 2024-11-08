# sketch2face3D


### Getting Start
```
git clone --recurse-submodules https://github.com/hailsong/sketch2face3D.git
```

It will recursively clone the repository of pix2pix3D, which is our baseline.
We further update its key features by inheriting and re-packaging.

```
conda env create -f environment.yml
conda activate sketch2face3d
```

### Experiments
We are preparing training code with conditions below.
- Baseline
    - (A) Edge 데이터 기반으로 celebA 학습시킨거
    - (B) Edge2Mask Unet 뒤에 연결한 pix2pix3D를 학습시킨거
    - (C) Edge2Mask Unet 중간 hidden layer와 style vector 공유
- 어디부터 어디까지 학습시킬지?
    - A같은 경우, **그냥 처음부터 (A1)**
    - B는 **Unet 구조랑 뒷부분 아예 따로 학습 (B1)** / 아예 처음부터 같이 할수도 / **뒷부분만 pre-trained (B2)**
    - C는 아예 처음부터 같이 할수도 / **뒷부분만 pre-trained (C1)**

Training code for each condition will be set inside of ./train_scripts. Condition A1 will be the first one. Good luck Better :)


### Goal
Sketch to 3D Modeling (of face dataset)

### TODO List
- [x] pix2pix3D inference demo - Hail
- [x] pix2pix3D training code check (with CelebA mask data) - Hail
- [x] Making Edge dataset for CelebA - Soomin
- [ ] pix2pix3D training code check (with CelebA edge data) - Hail
- [ ] Building U-net based Edge2Mask NN
- [ ] Building U-net based Edge2Mask NN (By sharing style vector of pix2pix3D Encoder)
- [ ] Using 3D Gaussian Splatting for Neural Rendering part
- [ ] Using Diffusion based methods for the original GAN part

### Issues
- Lpips 설치시 pytorch 버전 종속성 이슈

    ```
    pip install lpips --no-deps
    ```

- torchvision 설치시 pytorch 버전 종속성 이슈
    ```
    pip install torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html --no-deps
    ```
