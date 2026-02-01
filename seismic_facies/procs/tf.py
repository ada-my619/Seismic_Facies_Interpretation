import random
import torch
import torchvision.transforms.functional as TF

class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class RandomRotate:
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, x, y):
        angle = random.uniform(-self.degrees, self.degrees)
        # rotate x (float image)
        x = TF.rotate(x, angle)
        # rotate y (label) as image with nearest interpolation
        y = TF.rotate(y.unsqueeze(0).float(), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()
        return x, y


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x, y):
        if random.random() < self.p:
            x = TF.hflip(x)
            y = TF.hflip(y)
        return x, y


class GaussianBlurXOnly:
    def __init__(self, kernel_size=3, sigma=None, p=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = p

    def __call__(self, x, y):
        if random.random() < self.p:
            x = TF.gaussian_blur(x, kernel_size=self.kernel_size, sigma=self.sigma)
        return x, y


class AddNoiseXOnly:
    def __init__(self, std=0.01, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x, y):
        if random.random() < self.p:
            x = x + torch.randn_like(x) * self.std
        return x, y
