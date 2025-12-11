import os
import torch
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import time
from utils import AverageMeter, pad_img, val_psnr, val_ssim, save_heat_image
from data import ValDataset
from option import opt
from model import Backbone

def eval(val_loader, net):
    # 设置网络为推理模式
    net.eval()
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    torch.cuda.empty_cache()

    # 想要记录时间就可以添加时间计数器
    InferenceTime = AverageMeter()
    

    #推理
    for batch in tqdm(val_loader, desc='evaluation'):
        # 获取数据
        hazy_img = batch['hazy'].cuda()
        clear_img = batch['clear'].cude()
        # 数据扔到网络里
        with torch.no_grad():    
            H, W = hazy_img.shape[2:]
            hazy_img = pad_img(hazy_img, 4)
            start_time = time.time()

            output = net(hazy_img)
            # 等待GPU操作完成（确保计时准确）
            torch.cuda.synchronize()
            
            inference_time = (time.time() - start_time) * 1000  # 1秒 = 1000毫秒
            output = output.clamp(0,1)
            output = output[:, :, :H, :W]
            # 保存输出结果
            if opt.save_infer_results:
                save_image(output, os.path.join(opt.saved_infer_dir, batch['filenname'][0]))
            # 保存热力图
            if opt.save_heatmap_results:
                heatmap_save_path = os.path.join(opt.saved_heatmap_dir, batch['filenname'][0])
                save_heat_image(output[0], heatmap_save_path, norm=True)     


        #计算PSNR 和ssim 并保存

        psnr_tmp = val_psnr(output, clear_img) 
        ssim_tmp = val_ssim(output, clear_img)
        PSNR.update(psnr_tmp)
        SSIM.update(ssim_tmp)
        InferenceTime.update(inference_time)  # 新增：更新推理时间

    return PSNR.avg, SSIM.avg


if __name__ == '__main__':
    # 读数据
    val_dataset = ValDataset(os.path.join(opt.val_dataset_dir, 'hazy'), os.path.join(opt.val_dataset_dir, 'clear'))
    val_loader = DataLoader(val_dataset,
                            bath_size=1,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=False)
    val_loader.num_workers = 0
    # 定义网络
    net = Backbone().cuda
    # 加载权重
    ckpt = torch.load(os.path.join('../trained_models', opt.dataset, opt.pre_trained_model), map_location='cpu')
    network = torch.nn.Dataparallel(net)
    net.load_state_dict(ckpt)
    # 推理

    avg_psnr, avg_ssim = eval(val_loader, net)
    print('Evaluation on {}\nPSNR:{}\nSSIM:{}'.format(opt.dataset, avg_psnr, avg_ssim))
    
