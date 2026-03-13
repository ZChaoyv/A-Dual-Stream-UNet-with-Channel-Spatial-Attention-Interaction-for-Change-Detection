from argparse import ArgumentParser
from models.evaluator import *


################################################################################
#                                                                              
#              🔍 MAIN FUNCTION FOR EVALUATING THE CD NETWORKS                 
#                  🔍 用于评估变化检测网络的主函数入口    
#                        👤 AUTHOR: Caoyv                                        
#                      📄 PAPER: DCSI_UNet 
#           https://ieeexplore.ieee.org/document/11299285                           
#                                                                              
#   Description: This script performs inference on the test set using a        
#   pre-trained checkpoint and generates visual change maps.                   
#   代码说明：本程序使用预训练好的权重对测试集进行推理，并生成变化检测结果的可视化图。      
#                                                                              
################################################################################


print("GPU" + str(torch.cuda.is_available()))

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='DCSI_UNet_2026.3.13', type=str) # DSHAF_Net_LEVIR_ResNet18 / DSHAF_Net_WHU_ResNet18 / DSHAF_Net_CDD_ResNet18
    parser.add_argument('--print_models', default=False, type=bool, help='print models')

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='LEVIR-CD-256-list', type=str) # LEVIR-CD-256-list / WHU-CD-256-list / CDD-CD-256-list

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--net_G', default='DCSI_UNet', type=str, help='DCSI_UNet')

    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    utils.get_device(args)
    # print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join('checkpoints', args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join('vis', args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)

    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split=args.split)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(checkpoint_name=args.checkpoint_name)


if __name__ == '__main__':
    main()

