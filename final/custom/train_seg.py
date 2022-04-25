from .model_seg import FCN, save_model, load_model
import torch
import torch.utils.tensorboard as tb
import numpy as np
from .utils import load_dense_data
from . import dense_transforms
import random
import torch.nn.functional as F
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import torch.nn.functional as F
import numpy as np
from .utils import load_dense_data, ConfusionMatrix
from . import dense_transforms
import torch.utils.tensorboard as tb
from os import path
import random
import numpy as np
from torchvision import transforms

#https://amaarora.github.io/2020/06/29/FocalLoss.html
#https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/16
class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):
        patterns = target.shape[0]
        tot = 0
        for b in range(patterns):
            ce_loss = F.cross_entropy(input[b:b+1,], target[b:b+1],reduction=self.reduction,weight=self.weight)
            pt = torch.exp(-ce_loss)
            focal_loss = (1 - pt) ** self.gamma * ce_loss
            tot = tot + focal_loss
        return tot/patterns

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

    model =FCN()
    # train_logger, valid_logger = None, None
    # if args.log_dir is not None:
    #     train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    #     valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    # summary(model,(3,128,128))
    
    """
    Your code here.
    Finish logging the dummy loss and accuracy
    Log the loss every iteration, the accuracy only after each epochtorch.optim.SGD

    Make sure to set global_step correctly, for epoch=0, iteration=0: global_step=0
    Call the loss 'loss', and accuracy 'accuracy' (no slash or other namespace)
    """
    if args.continue_training:
        from os import path
        model.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % args.model)))
    

    #Add transformations for Data Augmentation
    #https://medium.com/analytics-vidhya/transforming-data-in-pytorch-741fab9e008c

    batch_size =args.batch_size  #an example images 128x96
    transform = dense_transforms.Compose([
        dense_transforms.RandomHorizontalFlip(),
        dense_transforms.ToTensor()
        # dense_transforms.Normalize([0.2649056  ,0.26406983 ,0.2644049 ], [0.22418422, 0.22388573 ,0.22410716]) #[0.2649056  0.26406983 0.2644049 ] [0.22418422 0.22388573 0.22410716]

     ] )
    

    batch_size =args.batch_size
    training_data = 21000
    steps = 10
    q=int( training_data /batch_size)
    lr = args.lr
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1) #every 10k steps, every 2000
    # optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max =q,eta_min=0.000003)

    

    training = load_dense_data('image_data',batch_size=batch_size,  transform= transform)
    global_step = 0
    torch.manual_seed(33)
    random.seed(33)
    np.random.seed(33)

    means_all, stds_all, means_all_plain, stds_all_plain  =[],[],[],[]
    means, stds = [], []

    for epoch in range(args.epochs):
        # model.train()
        accu, iou, loss_= [ ], [ ], [ ]

        for X, y in training:
            X_plain =X
            print(X)
            X = X.to(device)
            y =( y.long()).to(device)
            pred_y = model(X).to(device)
            means = np.mean(X.detach().cpu().numpy(), axis=(0, 2, 3))
            stds =  np.std(X.detach().cpu().numpy(),  axis=(0, 2, 3))
            means_plain = np.mean(X_plain.detach().cpu().numpy(), axis=(0, 2, 3))
            stds_plain =  np.std(X_plain.detach().cpu().numpy(),  axis=(0, 2, 3))
#             #loss
            # class_weights = [1/(5*0.52683655), 1/(5*0.02929112), 1/(5*0.4352989), 1/(5*0.0044619), 1/(5*0.00411153)]
            # class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
            #https://androidkt.com/how-to-use-class-weight-in-crossentropyloss-for-an-imbalanced-dataset/
            # l = torch.nn.CrossEntropyLoss()
            # wl =torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
            fl= FocalLoss(reduction='mean', gamma =0.25) #weight=class_weights#change gamma 0.25
            loss = fl(pred_y, y).to(device)#)cross_entropy2d(pred_y, y)
             #loggin
            train_logger.add_scalar('loss', float(loss), global_step=global_step)
            # optimizer.zero_grad()
           
            #step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            #loggin
            cMat_training = ConfusionMatrix()
            cMat_training.add(pred_y.argmax(1), y)

            loss_.append(loss.detach().cpu().numpy())
            accu.append(cMat_training.global_accuracy.detach().cpu().numpy())
            iou.append(cMat_training.iou.detach().cpu().numpy())
            global_step += 1

            means_all.append(means)
            stds_all.append(stds)
            means_all_plain.append(means_plain)
            stds_all_plain.append(stds_plain)


        avgloss = np.mean(loss_)
        avgaccu = np.mean(accu)
        avgiou = np.mean(iou)
        train_logger.add_scalar('training accuracy',avgaccu, global_step=global_step)
        train_logger.add_scalar('training iou',avgiou, global_step=global_step)
        train_logger.add_scalar('loss',avgloss, global_step=global_step)
        log(train_logger, X, y,pred_y, global_step)
        

        # model.eval()

        save_model(model)
   

    means_mean = np.mean(means_all, axis=0)
    std_mean = np.mean(stds_all, axis=0)
    means_mean_plain = np.mean(means_all_plain, axis=0)
    std_mean_plain = np.mean(stds_all_plain, axis=0)
    print(means_mean, std_mean)
    print(means_mean_plain, std_mean_plain)
        
        

    
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir', default = 'temp')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help = 'batch size')
    parser.add_argument('-e', '--epochs', type=int , default=10, help = '# of epochs')
    parser.add_argument('-p', '--lr', type=float , default=0.003, help = 'optimization parameter for lr sgd')
    parser.add_argument('-c', '--continue_training', action='store_true')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    train(args)
    


