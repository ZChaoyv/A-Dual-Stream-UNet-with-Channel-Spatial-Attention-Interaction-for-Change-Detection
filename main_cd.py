from argparse import ArgumentParser
from models.trainer import *


################################################################################
#                                                                              
#              🚀 MAIN FUNCTION FOR TRAINING THE CD NETWORKS                   
#                  🚀 用于训练变化检测网络的主函数入口         
#                       👤 AUTHOR: Caoyv                                        
#                      📄 PAPER: DCSI_UNet 
#           https://ieeexplore.ieee.org/document/11299285                          
#                                                                              
#   Description: This script handles the full pipeline, including training      
#   on the training set and validation on the validation/test set.             
#   代码说明：本程序包含完整流程，涵盖了在训练集上的训练以及在验证/测试集上的评估。        
#                                                                              
################################################################################


print("GPU" + str(torch.cuda.is_available()))

def train(args):
    dataloaders = utils.get_loaders(args)
    model = CDTrainer(args=args, dataloaders=dataloaders)
    model.train_models()


def test(args):
    from models.evaluator import CDEvaluator
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split='test')
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models()

if __name__ == '__main__':

    # information collection
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='DCSI_UNet_2026.3.13', type=str) # DSHAF_Net_LEVIR_ResNet18 / DSHAF_Net_WHU_ResNet18 / DSHAF_Net_CDD_ResNet18
    parser.add_argument('--checkpoint_root', default='checkpoints', type=str)

    # data
    parser.add_argument('--num_workers', default= 8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR-CD-256-list', type=str) # LEVIR-CD-256-list  WHU-CD-256-list  CDD-CD-256-list

    parser.add_argument('--batch_size', default=48, type=int)
    parser.add_argument('--split', default="train", type=str)
    parser.add_argument('--split_val', default="val", type=str)

    parser.add_argument('--img_size', default=256, type=int)
 
    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='DCSI_UNet', type=str, help='DCSI_UNet') # DSHAFNet
    parser.add_argument('--loss', default='ce', type=str, help='ce | ce + dice')

    # optimizer
    parser.add_argument('--optimizer', default='sgd', type=str)
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--max_epochs', default=200, type=int)
    parser.add_argument('--lr_policy', default='linear', type=str, help='linear | step | static')
    parser.add_argument('--lr_decay_iters', default=200, type=int)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 of adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of adam optimizer')

    args = parser.parse_args()
    utils.get_device(args)
    # print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoint_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    train(args)

    test(args)
