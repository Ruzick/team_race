import torch
import torch.nn.functional as F


# def spatial_argmax(logit):
#     """
#     Compute the soft-argmax of a heatmap
#     :param logit: A tensor of size BS x H x W
#     :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
#     """
#     weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
#     return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
#                         (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=1):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_cls = F.max_pool2d(heatmap[None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)
    return torch.tensor([[ int(l) % heatmap.size(1), int(l) // heatmap.size(1)]
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score])


class Planner(torch.nn.Module):
    def __init__(self, channels=[16, 32, 32, 32]):
        super().__init__()

        conv_block = lambda c, h: [torch.nn.BatchNorm2d(h), torch.nn.Conv2d(h, c, 5, 2, 2), torch.nn.ReLU(True)]

        h, _conv = 3, []
        for c in channels:
            _conv += conv_block(c, h)
            h = c

        self._conv = torch.nn.Sequential(*_conv, torch.nn.Conv2d(h, 1, 1))
        # self.classifier = torch.nn.Linear(h, 2)
        # self.classifier = torch.nn.Conv2d(h, 1, 1)

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        return extract_peak(x[:, 0])
        # return self.classifier(x.mean(dim=[-2, -1]))


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


# if __name__ == '__main__':
#     from .controller import control
#     from .utils import PyTux
#     from argparse import ArgumentParser


#     def test_planner(args):
#         # Load model
#         planner = load_model().eval()
#         pytux = PyTux()
#         for t in args.track:
#             steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
#             print(steps, how_far)
#         pytux.close()


#     parser = ArgumentParser("Test the planner")
#     parser.add_argument('track', nargs='+')
#     parser.add_argument('-v', '--verbose', action='store_true')
#     args = parser.parse_args()
#     test_planner(args)
