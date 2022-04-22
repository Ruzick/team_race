from .model import Detector, save_model, load_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .runner import load_detection_data
from . import dense_transforms
import random


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def log(logger, img, label, pred, global_step):
    """
    logger: train_logger/valid_logger
    img: image tensor from data loader
    label: ground-truth aim point
    pred: predited aim point
    global_step: iteration
    """
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    fig, ax = plt.subplots(1, 1)
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    ax.add_artist(plt.Circle(WH2*(label[0].cpu().detach().numpy()+1), 2, ec='g', fill=False, lw=1.5))
    ax.add_artist(plt.Circle(WH2*(pred[0].cpu().detach().numpy()+1), 2, ec='r', fill=False, lw=1.5))
    logger.add_figure('viz', fig, global_step)
    del ax, fig

def train(args):

    from os import path
    model = Detector()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    
    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epochtorch.optim.SGD
    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    
    from os import path
        # model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))
    

    #Add transformations for Data Augmentation
    #https://medium.com/analytics-vidhya/transforming-data-in-pytorch-741fab9e008c

    batch_size =args.batch_size  #an example images 128x96
    hue = args.hue
    cont = args.cont
    sat = args.sat

    transform = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        # dense_transforms.ColorJitter( 0.9, 0.9, 0.9, hue=hue),#)contrast=0.5, saturation=0.9, hue=0.1), ##low in contrast and brightness for colorjitter #changed from .5 to .2
        dense_transforms.ToTensor()
        # dense_transforms.Normalize([0.2649056  ,0.26406983 ,0.2644049 ], [0.22418422, 0.22388573 ,0.22410716]) #[0.2649056  0.26406983 0.2644049 ] [0.22418422 0.22388573 0.22410716]
     ] )
    


    lr = args.lr
    step_size = args.opt_step_size
    gamma2 = args.opt_gamma

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma2) #every 10k steps
    
    global_step = 0
    torch.manual_seed(33)
    random.seed(33)
    np.random.seed(33)


    epoch = 0
    class_weights = args.list_o_w #[0.1160526, 0.7707894, 0.11736842] #detected ot not
    class_weights=torch.FloatTensor(class_weights).view(1,3,1,1).to(device)
    loss_all =[]
    # model = load_model().to(device)
    # if args.continue_training:
    #     model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'det.th')))
    model.train()
    for epoch in range(args.epochs):
        loss_= [ ]
        for image, peak_loc  in load_detection_data('ff_data/train',batch_size=64,  num_workers = 1,transform= transform):
            print(peak_loc)
            X = image.to(device)
            peak_pred  = model.detect(X)
            print(peak_pred)
            peak_pred = peak_pred.to(device)
            peak_loc =( peak_loc).to(device)
            delta = args.delta
            huber = torch.nn.HuberLoss(reduction = 'mean', delta = delta).to(device)           
            fl_huber = huber(peak_pred,peak_loc).to(device)

            loss = fl_huber *100

            loss_.append(loss.detach().cpu().numpy())



            #loggin
            log(train_logger, image, peak_loc, peak_pred, global_step)
            train_logger.add_scalar('loss', np.mean(loss_), global_step)

           
            #step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            mean_l = np.mean(loss_)
            global_step += 1

        loss_all.append( np.mean(loss_))
  
        print(f'epoch {epoch} loss { np.mean(loss_)} learning rate {scheduler.get_last_lr()[0]} ')
        save_model(model)

    

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir', default = 'temp')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help = 'batch size')
    parser.add_argument('-e', '--epochs', type=int , default=10, help = '# of epochs')
    parser.add_argument('-p', '--lr', type=float , default=0.003, help = 'optimization parameter for lr sgd')
    # parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-a', '--alpha',type=float, help='alpha', default=2)
    parser.add_argument('-g', '--gamma',type=float,help='gamma',default=0.25)
    parser.add_argument('-d', '--delta',type=float,help='delta_loss_hubber',default=1.0)
    parser.add_argument('-s', '--opt_step_size',type= int,  help='step_size',default=10000)
    parser.add_argument('-og', '--opt_gamma',type=float, help='gamma for opt', default=0.1)
    parser.add_argument('-l','--list_o_w', nargs='+', type = float, help='list_of_weights sep by space', default = [ 0.75 ,1 ,1])#1/(5*0.02929112), 1/(5*0.0044619), 1/(5*0.00411153)])
    parser.add_argument('-hue', '--hue',type=float, help='hue', default=0.5)
    parser.add_argument('-cont', '--cont',type=float, help='cont', default=0.9)
    parser.add_argument('-sat', '--sat',type=float, help='sat', default=0.9)
    parser.add_argument('-conti', '--continue_training',type=bool, help='sat', default=False)
    args = parser.parse_args()

    from os import path
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    train(args)
