# Source: https://github.com/pytorch/vision/blob/master/references/segmentation/transforms.py
import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, *args):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            args = tuple(np.array([-point[0], point[1]],
                         dtype=point.dtype) for point in args)
        return (image,) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, *args):
        for t in self.transforms:
            image, *args = t(image, *args)
        return (image,) + tuple(args)


class ColorJitter(T.ColorJitter):
    def __call__(self, image, *args):
        return (super().__call__(image),) + args


class ToTensor(object):
    def __call__(self, image, *args):
        return (F.to_tensor(image),) + args


class ToHeatmap(object):
    def __init__(self, radius=2):
        self.radius = radius

    def __call__(self, image, *dets):
        peak = detections_to_heatmap(dets, image.shape[1:], radius=self.radius)
        return image, peak


def detections_to_heatmap(dets, shape, radius=2, device=None):
    with torch.no_grad():
        # size = torch.zeros((2, shape[0], shape[1]), device=device)
        peak = torch.zeros((len(dets), shape[0], shape[1]), device=device)
        for i, det in enumerate(dets):
            if len(det):
                det = torch.tensor(det.astype(
                    float), dtype=torch.float32, device=device)
                cx, cy = (det[:, 0] + det[:, 2] - 1) / \
                    2, (det[:, 1] + det[:, 3] - 1) / 2
                x = torch.arange(shape[1], dtype=cx.dtype, device=cx.device)
                y = torch.arange(shape[0], dtype=cy.dtype, device=cy.device)
                gx = (-((x[:, None] - cx[None, :]) / radius)**2).exp()
                gy = (-((y[:, None] - cy[None, :]) / radius)**2).exp()
                gaussian, id = (gx[None] * gy[:, None]).max(dim=-1)
                # mask = gaussian > peak.max(dim=0)[0]
                # det_size = (det[:, 2:] - det[:, :2]).T / 2
                # size[:, mask] = det_size[:, id[mask]]
                peak[i] = gaussian
        return peak


class CenterToHeatmap(object):
    def __init__(self, radius=2) -> None:
        self.radius = radius

    def __call__(self, image, *dets):
        peak = centers_to_heatmap(dets, image.shape[1:], radius=self.radius)
        return image, peak


def centers_to_heatmap(dets, shape, radius=2, device=None):
    with torch.no_grad():
        peak = torch.zeros((len(dets), shape[0], shape[1]), device=device)
        for i, det in enumerate(dets):
            if len(det):
                det = torch.tensor(det.astype(
                    float), dtype=torch.float32, device=device)

                # Note: There are some issues with the dimensions of labels with one entry
                if det.ndim == 1:
                    det = torch.unsqueeze(det, dim=0)

                cx, cy = det[:, 1], det[:, 0]
                x = torch.arange(shape[1], dtype=cx.dtype, device=cx.device)
                y = torch.arange(shape[0], dtype=cy.dtype, device=cy.device)
                gx = (-((x[:, None] - cx[None, :]) / radius)**2).exp()
                gy = (-((y[:, None] - cy[None, :]) / radius)**2).exp()
                gaussian, id = (gx[None] * gy[:, None]).max(dim=-1)
                peak[i] = gaussian
        return peak
