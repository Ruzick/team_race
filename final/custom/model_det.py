import torch
import torch.nn.functional as F
import pystk

def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks // 2, stride=1)[0, 0]
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
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32, 64, 128], n_class=1, kernel_size=3, use_skip=True):
        """
           Your code here.
           Setup your detection network
        """
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
        self.classifier = torch.nn.Conv2d(c, n_class, 1) #changed n_class to puck only
        # self.size = torch.nn.Conv2d(c, 2, 1)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        z = (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)
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
        return self.classifier(z)#, self.size(z)

    def detect(self, image, player, **kwargs):
        import numpy as np
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image    ....3x300x400
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.

                 Note: Each frame will use this function to detect the puck, depending on the segmentation it will be able
                 to detect it at different depths; the segmentation could be changed to detect smaller objects
                 in order to improve this. The input data only contains information of when the puck is in front of the kart.

        """
        hm = self.forward(image)#.unsqueeze(0)) #image[None] shape is then (1 x 1 x h x w ) classes are puck and player
        all_lists= []
        for channel in range (hm.size(dim= 1)): #only one channel for now (puck)
            peaks_per_object = []
            current_list = extract_peak(hm[0][channel], max_det = 30)
            if len(current_list) == 0:
                if channel ==0:
                    print("no object detected", current_list,channel)
                    current_list =[0,0]
                #puck is somewhere in the front but not caught by segmentation
            else: 
            #get the max peak value, so it doenst think something else is a puck
                print("puck detected", channel)
                puck = max(current_list, key=lambda x: x[0])
                cx = puck[1]
                cy = puck[2]
            #     current_list =[cx,cy]
            # for each_object in current_list:
            #     print(each_object, 'each object')
            #     peaks_per_object.append(int(each_object[0]),int(each_object[1])) 
            

        #     all_lists.append(peaks_per_object) #contains both 
        # print(puck)

        #convert to normalize image

            proj = np.array(state.players[i].camera.projection).T
            view = np.array(state.players[i].camera.view).T
            p = proj @ view @ np.array(list(state.soccer.ball.location) + [1])
            aim = np.clip(np.array([p[0] / p[-1], -p[1] / p[-1]]), -1, 1) #image coordinates of puck

        return  [cx,cy]#all_lists

        # cls, size = self.forward(image[None])
        # size = size.cpu()
        # return [[(s, x, y, float(size[0, 0, y, x]), float(size[0, 1, y, x]))
        #          for s, x, y in extract_peak(c, max_det=30, **kwargs)] for c in cls[0]]


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
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

    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    fig, axs = subplots(3, 4)
    model = load_model()
    for i, ax in enumerate(axs.flat):
        im, puck = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        score, cx, cy= model.detect(im.to(device))
        print('x, y', cx, cy )
        ax.add_patch(patches.Circle((cx, cy), radius=max(2  / 2, 0.1), color='rgb'[0]))
        ax.axis('off')
    show()