from pathlib import Path
import torch
import torch.nn.functional as F
import pystk
from torchvision import transforms

#

def extract_peak(heatmap, max_pool_ks=7, min_score=0.4, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
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


def Block(n_input_channels, n_output_channels):
    return torch.nn.Sequential(
                    torch.nn.Conv2d(n_input_channels,n_output_channels,kernel_size=(1,3), padding = 0),
                    torch.nn.Conv2d(n_output_channels,n_output_channels,kernel_size=(3,1), padding = 1),
                    torch.nn.BatchNorm2d(n_output_channels),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1)

    )  
def BlockUp(n_input_channels, n_output_channels):
    return torch.nn.Sequential(
                    torch.nn.Conv2d(n_input_channels,n_output_channels,kernel_size=(1,3), padding = 0),
                    torch.nn.Conv2d(n_output_channels,n_output_channels,kernel_size=(3,1), padding = 1),
                    torch.nn.BatchNorm2d(n_output_channels),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1)
    )  

class Detector(torch.nn.Module):          
    def __init__(self, layers= [32,64,128]):#,256]): 16, 32, 64, 128 n_class=3, kernel_size=3, use_skip=True)
        super().__init__()
    #     """
    #     Your code here.
    #     Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
    #     Hint: Use up-convolutions
    #     Hint: Use skip connections
    #     Hint: Use residual connections
    #     Hint: Always pad by kernel_size / 2, use an odd kernel_size
    #     """
        self.conv1 = Block(3,layers[0]) # 1 to 32 we are getting heatmap per channel now
        self.conv2 = Block(layers[0], layers[1]) #32 to 64
        self.conv3 = Block(layers[1], layers[2]) #64 to 128
        # self.conv4 = Block(layers[2], layers[3]) #128 to 256

        self.maxpool = torch.nn.MaxPool2d(kernel_size=4, padding=1, stride=2, dilation=1) #(H+2p-dil*(kern-1)-1)/str  +1  or dilation =3 kernel_size=2
        self.upscale = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) 

        # self.up3 =BlockUp(layers[3], layers[2]) #256 to 128

        self.up2 = BlockUp(layers[2], layers[1]) #128 to 64
        self.up1 =BlockUp(layers[1], layers[0]) #64 to 32
        #layers= [32,64,128,256]
        self.up2cat = BlockUp(layers[2]*2, layers[1]) #128+128  to 64
        self.up1cat =BlockUp(layers[1]*2, layers[0]) #128  to 32


        self.classifier = torch.nn.Conv2d(layers[0], 3, kernel_size=1) #2 diff classes
        self.sigmoid = torch.nn.Sigmoid()
        self.input_mean = torch.Tensor([0.2788, 0.2657, 0.2629])
        self.input_std = torch.Tensor([0.2064, 0.1944, 0.2252])


    def forward(self, x):

        b, c, h, w = x.shape
        x =  (x - self.input_mean[None, :, None, None].to(x.device)) / self.input_std[None, :, None, None].to(x.device)

        target_image_size =(320,320)# (160,160)  # as an example images 128x96
        
        transform=transforms.Compose([
            transforms.Pad((60,60,60,60),padding_mode= 'constant',fill=0),
            transforms.CenterCrop(target_image_size)
         ])
        # [32,64,128])
        #************************************************************************************
        #          ENCODER                      3x320x320
        #*******************************************************************************
        #x  torch.Size([128, 3, 96, 128])
        x = transform(x)  #          3x320x320
        #first convolution
        conv1 =  self.conv1(x)  #     32x320x320
        x= self.maxpool(conv1)  #    32 x 160 x 160
        #2ndconvolution
        conv2 =  self.conv2(x)     #   64x160x160
        x= self.maxpool(conv2)   # 64x 80x80

        #3rdconvolution
        conv3 =  self.conv3(x)     #128x80x80
        x= self.maxpool(conv3)  #  128x16x12             128x16x16           128x10x10                            128x40x40             

        #****************************************************************************************
        #           DECODER 1 (USING SKIPPED CONNECTIONS (Addition) +  UPSCALE) 128x16x16 
        #*******************************************************************************************

        #upconvolution 1
        x=  self.upscale(x)  #  128x40x40
        x = x+ conv3
        x = self.up2(x) #    64x40x40 


        #upconvolution2
        x=  self.upscale(x) #   64x80x80 
        x = x+conv2 #    
        x= self.up1(x) #      32x80x80
        
        #upconvolution3

        x= self.upscale(x) #     32x160x160
        x = x+conv1  # x = x+conv2

        y_pred = self.classifier(x)
        y_pred=transforms.CenterCrop((h,w))( y_pred )
        #y_pred =self.sigmoid(y_pred)
        
        return y_pred

    def detect(self, image, max_det=4):
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
        for channel in range (hm.size(dim= 1)): #channel 0 : puck , channel 1: kart, channel 3, goal
            peaks_per_object = []
            if channel == 0:
                current_list = extract_peak(torch.sigmoid(hm[0][channel]), max_det = 1)
            if channel == 1:
                current_list = extract_peak(torch.sigmoid(hm[0][channel]), max_det = 3)
            if channel == 2:
                current_list = extract_peak(torch.sigmoid(hm[0][channel]), max_det = 1)
            
            peaks_per_object.append(current_list)       

            for each_object in current_list:

                peaks_per_object.append(    [ int(each_object[1]),int(each_object[2])])
            all_lists.append(peaks_per_object)


        return  all_lists


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