import torch
import numpy as np
import random
from .model_det import Detector, save_model, load_model
from .utils import load_detection_data
from . import dense_transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard as tb
from os import path
import torchvision
import torch.nn.functional as F




def log(logger, imgs, lbls, logits, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    lbls: semantic label tensor
    logits: predicted logits tensor
    global_step: iteration
    """
    logger.add_image('image', imgs[0], global_step)
    logger.add_image('label', np.array(dense_transforms.label_to_pil_image(lbls[0].cpu()).
                                            convert('RGB')), global_step, dataformats='HWC')
    logger.add_image('prediction', np.array(dense_transforms.
                                                label_to_pil_image(logits[0].argmax(dim=0).cpu()).
                                                convert('RGB')), global_step, dataformats='HWC')

def train(args):#test_logging(train_logger, valid_logger):

    model =Detector()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    if args.continue_training:
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'planner.th')))
    model.train()
    
    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epochtorch.optim.SGD
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    
    from os import path

    batch_size =args.batch_size  #an example images 128x96
    hue = args.hue
    cont = args.cont
    sat = args.sat

    transform = dense_transforms.Compose([
        # dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor()
        # dense_transforms.ToHeatmap()
        # dense_transforms.Normalize([0.2649056  ,0.26406983 ,0.2644049 ], [0.22418422, 0.22388573 ,0.22410716]) #[0.2649056  0.26406983 0.2644049 ] [0.22418422 0.22388573 0.22410716]
     ] )
    


    lr = args.lr
    step_size = args.opt_step_size
    gamma2 = args.opt_gamma
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # w = torch.as_tensor(DENSE_CLASS_DISTRIBUTION)**(-args.gamma)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma2) #every 10k steps
    

    training =load_detection_data('image_data',batch_size=batch_size,  transform= transform)
    global_step = 0
    torch.manual_seed(33)
    random.seed(33)
    np.random.seed(33)

    epoch = 0
    class_weights = args.list_o_w #[0.1160526, 0.7707894, 0.11736842] #detected ot not
    class_weights=torch.FloatTensor(class_weights).view(1,3,1,1).to(device)#torch.FloatTensor(class_weights) #torch.FloatTensor([weight_karts, weight_bombs, weight_pickups]).view(1,3,1,1).to(device)
    posol = []
    loss_all =[]
    # model.train()
    for epoch in range(args.epochs):

        loss_= [ ]
        pos = []
        for image, peak_hm  in load_detection_data('image_data',batch_size=batch_size,  transform= transform):

            positives = torch.sum(peak_hm.sigmoid())
            elements = torch.numel(peak_hm)
 
            X = image.to(device)
            peak_pred , size_pred = model(X)

            peak_pred = peak_pred.to(device)
            peak_hm =( peak_hm).to(device)

            alpha= args.alpha
            gamma= args.gamma

            bce = torch.nn.BCEWithLogitsLoss(reduction = 'none' ).to(device)                       
            fl_bce = bce(peak_pred,peak_hm).to(device)

            shifted_inputs = torch.sigmoid(- gamma * (peak_pred * (2 *peak_hm - 1)))
            # # # # # # https://github.com/facebookresearch/fvcore/blob/main/fvcore/nn/focal_loss.py
            # # # # # # loss = -(F.logsigmoid(shifted_inputs)) / gamma
            loss = (fl_bce * shifted_inputs) / gamma
            if alpha >= 0:
                alpha_t = alpha * peak_hm+ (1 - alpha) * (1 - peak_hm)
                loss *= alpha_t
            # pos = torch.sigmoid(shifted_inputs)
            loss = loss.mean()/shifted_inputs.mean() #/ positives_per_channel.mean()#.mean()/pos.mean() #.mean()/ positives_per_channel.mean() #pos.sum()
            loss_.append(loss.detach().cpu().numpy())



            #loggin
            log(train_logger, image, peak_hm, peak_pred, global_step)
            train_logger.add_scalar('loss', np.mean(loss_), global_step)

           
            #step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            mean_l = np.mean(loss_)
            global_step += 1
        loss_all.append(mean_l)
        # pos_mean  = np.mean(pos)
        print(f'epoch {epoch} loss {mean_l } learning rate {scheduler.get_last_lr()[0]} ')
        save_model(model)



        

def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)
    
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir', default = 'temp')
    parser.add_argument('-b', '--batch_size', type=int, default=34, help = 'batch size')
    parser.add_argument('-e', '--epochs', type=int , default=10, help = '# of epochs')
    parser.add_argument('-p', '--lr', type=float , default=0.003, help = 'optimization parameter for lr sgd')
    # parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-a', '--alpha',type=float, help='alpha', default=2)
    parser.add_argument('-g', '--gamma',type=float,help='gamma',default=0.25)
    parser.add_argument('-s', '--opt_step_size',type= int,  help='step_size',default=10000)
    parser.add_argument('-og', '--opt_gamma',type=float, help='gamma for opt', default=0.1)
    parser.add_argument('-l','--list_o_w', nargs='+', type = float, help='list_of_weights sep by space', default = [ 0.75 ,1 ,1])#1/(5*0.02929112), 1/(5*0.0044619), 1/(5*0.00411153)])
    parser.add_argument('-hue', '--hue',type=float, help='hue', default=0.5)
    parser.add_argument('-cont', '--cont',type=float, help='cont', default=0.9)
    parser.add_argument('-sat', '--sat',type=float, help='sat', default=0.9)
    parser.add_argument('-conti', '--continue_training',type=bool, help='sat', default=False)
    # parser.add_argument('-o', '--optimizer', default ='optim.SGD(parameters, lr = args.lr)')
    args = parser.parse_args()
    # optimizer = eval(args.optimizer, {'parameters': model.parameters(), 'optim': torch.optim})
    from os import path
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    train(args)




