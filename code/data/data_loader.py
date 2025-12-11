import os, random
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, Resize

class TrainDataset(data.Dataset):
    # 获取目录
    def __init__(self, hazy_path, clear_path):
        super(TrainDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
    # 获取图像, 一次只会返回一对训练样本对， 但是会根据batch_size = 调用次数
    def __getitem__(self, index):
        # 读噪声图像和干净的图片
        # 
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0]
        
        
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')       
        

        # 裁剪
        crop_params = RandomCrop.get_params(hazy, [256,256])
        hazy = crop(hazy, *crop_params) # *crop_params：这里的*是 “解包” 操作，将crop_params元组中的 4 个参数（top, left, height, width）依次传递给crop函数。
        clear = crop(clear, *crop_params)
        # 旋转
        rotate_params = random.randint(0, 3) * 90
        hazy = rotate(hazy, *rotate_params)
        clear = rotate(clear, *rotate_params)

        # 转化为tensor
        to_tensor = ToTensor()

        # 返回tensor格式的数据
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return hazy, clear
    def __len__(self):
        return len(self.hazy_image_list)
    
class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)

        self.hazy_image_list.sort()
        self.clear_image_list.sort()

    def __getitem__(self, index):
        # 获取图片
            # 读取 一张hazy和一张clear
            #  
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0]    
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name) 
        # 按照RGB模式读取图片
        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        # 转换成tensor
        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)
        
        return hazy, clear, hazy_image_name
    
    def __len__(self):
        return len(self.hazy_image_list)
    
class ValDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = os.listdir(hazy_path)
        self.clear_image_list = os.listdir(clear_path)
        self.hazy_image_list.sort()
        self.clear_image_list.sort()
    def __getitem__(self, index):
        #读图像
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = hazy_image_name.split('_')[0]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)

        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).conver('RGB')


        # RGB格式给数据读出来
        #转tensor
        to_tensor = ToTensor()

        hazy = to_tensor(hazy)
        clear = to_tensor(clear)

        return {'hazy': hazy, 'clear': clear, 'filename': hazy_image_name}


    def __len__(self):
        return len(self.hazy_image_list)

