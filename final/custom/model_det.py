from pathlib import Path
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
from torchvision.ops import DeformConv2d


def extract_peak(heatmap, max_pool_ks=7, min_score=0.2, max_det=100):
    """
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """

    max_cls = F.max_pool2d(
        heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)
    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]


class Detector(torch.nn.Module):

    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(
                n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.c3 = torch.nn.Conv2d(
                n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(
                n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_class=3, kernel_size=3, use_skip=True):
        super().__init__()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        # Produce lower res output
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_class, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)
             ) / self.input_std[None, :, None, None].to(x.device)
        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)  # , self.size(z)

    def detect(self, image, max_det=4):
        """
           @image: 3 x H x W image    ....3x300x400
           @return: Three list of detections [(score, cx, cy), ...], one per class,
                    [('kart', 4), ('puck', 1), ('goal', 1)]
        """
        if len(image.shape) < 4:
            image = image[None, ...]

        hm = self.forward(image)

        detects_by_class = []
        for c in range(hm.size(1)):
            detects = extract_peak(torch.sigmoid(hm[0][c]), max_det=max_det)
            detects_by_class.append(detects)

        return detects_by_class


class DeformableDetector(torch.nn.Module):

    class DeformDown(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2) -> None:
            super().__init__()

            self.modulated = True
            self.deformable_groups = 1

            if self.modulated:
                # Use modulation
                offset_channels = 27
            else:
                # Use normal deconv
                offset_channels = 18

            # Downsample convolution
            self.conv1 = torch.nn.Conv2d(
                n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)

            # offset layer for DeformConv
            self.dconv2_offset = torch.nn.Conv2d(
                n_output, offset_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            # Deformable convolution
            self.dconv2 = DeformConv2d(
                n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)

            # offset layer for DeformConv
            self.dconv3_offset = torch.nn.Conv2d(
                n_output, offset_channels, kernel_size=kernel_size, padding=kernel_size // 2)
            # Deformable convolution
            self.dconv3 = DeformConv2d(
                n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)

            self.bn1 = torch.nn.BatchNorm2d(n_output)
            self.bn2 = torch.nn.BatchNorm2d(n_output)
            self.bn3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(
                n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):

            out = self.conv1(x)
            out = self.bn1(out)
            out = F.relu(out)

            if self.modulated:
                offset_mask = self.dconv2_offset(out)
                offset = offset_mask[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.dconv2(out, offset, mask)
            else:
                offset = self.dconv2_offset(out)
                out = self.dconv2(out, offset)

            out = self.bn2(out)
            out = F.relu(out)

            if self.modulated:
                offset_mask = self.dconv2_offset(out)
                offset = offset_mask[:, :18 * self.deformable_groups, :, :]
                mask = offset_mask[:, -9 * self.deformable_groups:, :, :]
                mask = mask.sigmoid()
                out = self.dconv2(out, offset, mask)
            else:
                offset = self.conv2_offset(out)
                out = self.dconv2(out, offset)

            out = self.bn2(out)
            out = F.relu(out + self.skip(x))
            return out

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_class=3, kernel_size=3, use_skip=True, crop_top=0) -> None:
        super().__init__()

        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])

        c = 3
        self.crop_top = crop_top
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' %
                            i, self.DeformDown(c, l, kernel_size, 2))
            c = l
        # Produce lower res output
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_class, 1)

    def forward(self, x):
        z = (x - self.input_mean[None, :, None, None].to(x.device)
             ) / self.input_std[None, :, None, None].to(x.device)
        if self.crop_top > 0:
            top = int(self.crop_top*x.size(-2))
            height = x.size(-2)-top
            width = x.size(-1)
            z = TVF.crop(z, top, 0, height, width)

        up_activation = []
        for i in range(self.n_conv):
            # Add all the information required for skip connections
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]
            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)
        return self.classifier(z)  # , self.size(z)

    def detect(self, x, max_det=0.4, min_score=0.2):
        if len(image.shape) < 4:
            image = image[None, ...]

        detects_by_class = []
        hm = self.forward(image)
        for c in range(hm.size(1)):
            detects = extract_peak(torch.sigmoid(
                hm[0][c]), max_det=max_det, min_score=min_score)
            detects_by_class.append(detects)
        return detects_by_class


def save_model(model, outpath=Path("./models"), outname=None):
    from torch import save
    filename = f"det.th"
    if outname is not None:
        filename = f"det_{outname}.th"

    outpath.mkdir(exist_ok=True)
    return save(model.state_dict(), outpath / filename)


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(
        path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('ff_data')
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    fig, axs = subplots(3, 4)
    model = load_model()
    for i, ax in enumerate(axs.flat):
        im, puck = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        score, cx, cy = model.detect(im.to(device))
        print('x, y', cx, cy)
        ax.add_patch(patches.Circle(
            (cx, cy), radius=max(2 / 2, 0.1), color='rgb'[0]))
        ax.axis('off')
    show()
