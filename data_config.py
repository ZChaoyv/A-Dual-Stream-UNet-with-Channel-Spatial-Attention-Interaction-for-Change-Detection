

################################################################################
#                                                                              
#              📂 DATA CONFIGURATION & PATH MANAGEMENT                         
#                  📂 数据集配置与路径管理中心     
#                        👤 AUTHOR: Caoyv                                        
#                      📄 PAPER: DCSI_UNet 
#           https://ieeexplore.ieee.org/document/11299285                                 
#                                                                              
#   Description: This class centralizes all dataset paths and preprocessing    
#   settings. It maps dataset aliases to their respective physical locations.   
#   代码说明：该类统一管理所有数据集路径与预处理设置，将数据集别名映射至物理存储路径。       
#                                                                              
################################################################################


class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR-CD-256-list':
            self.root_dir = r'/home/chaoyv/chaoyv_File/Datasets/LEVIR-CD-256-list'  # Your Path
        elif data_name == 'WHU-CD-256-list':
            self.root_dir = r'/home/chaoyv/chaoyv_File/Datasets/WHU-CD-256-list'
        elif data_name == 'CDD-CD-256-list':
            self.root_dir = r'/home/chaoyv/chaoyv_File/Datasets/CDD-CD-256-list'
             
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR-CD-256-list')


