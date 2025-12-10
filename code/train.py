import os, time, math
import numpy as np 
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torchvision.utils import save_iamge
from torch.utils.data import DataLoader

from logger import plot_loss_log, plot_psnr_log
from metric import psnr, ssim
from model import DiGPaRNet
from loss import ContrastLoss
from option_train import opt
from data.data_loader import TrainDataset, TestDataset

steps = opt.iters_per_epoch * opt.epochs
T = steps

def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr-end_lr) * (1+math.cos(t * math.pi / T))
    return lr

def train(net, loader_train, loader_test, optim, criterion):
    # 设计train的流程
    
    #需要记录的参数,到时方便可视化
    losses = []
    loss_log = {'L1': [], 'CR': [], 'total': []}
    loss_log_tmp = {'L1': [], 'CR': [], 'total': []}
    psnr_log = []

    ssims = []
    psnrs = []
    max_psnr = 0
    max_ssim = 0
    # 设置训练的一些参数
    start_step = 0
    
    # 将训练数据转化为迭代器
    loader_train_iter = iter(loader_train)
    # 开启训练
    for step in range(start_step + 1, step + 1):
        # 训练模式
        net.train()
        # 设置学习率，和学习率衰减函数
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
        
        # 读取数据
        x, y = next(loader_train_iter)
        x = x.to(opt.device)
        y = y.to(opt.device)

        # 输入需要复原的图片放到网络里，开网络流
        out = net(x)
        
        # 让网络输出和清晰图片 计算loss
        if opt.w_loss_L1 > 0:
            loss_L1 = criterion[0](out, y)
        if opt.w_loss_CR > 0:
            loss_CR = criterion[1](out, y, x)
        loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_CR * loss_CR
        # 反向传播更新网络参数
        loss.backward()                     # 计算梯度
        optim.step()                        # 优化器根据计算额梯度来更新模型参数
        optimizer.zero_grad()               # 将优化器中参数的梯度清零
        
        #记录损失，为了可视化
        losses.append(loss.item())
        loss_log_tmp['L1'].append(loss_L1.item())
        loss_log_tmp['CR'].append(loss_CR.item())
        loss_log_tmp['total'].append(loss.item())

        # 在控制台输出一下
        print(
            f'\rloss:{loss.item():.5f} | L1:{loss_L1.item():.5f} | CR:{opt.w_loss_CR * loss_CR.item():.5f} | step :{step}/{steps} | lr :{lr :.7f} | time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)

        # 每个epoch 完以后记录一下结果 和 操作  记录epoch 平均损失
        if step % len(loader_train) == 0:
            loader_train_iter = iter(loader_train)

        # 验证 - 训练多少个iter 然后验证一下
        # 定期验证，定期保存模型，记录关键指标  
        if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or (step > opt.finer_eval_step and (step - opt.finer_eval_step) % opt.iter_per_epoch == 0):
            if step > opt.finer_eval_step:
                epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // opt.iters_per_epoch
            else:
                epoch = int(step / opt.iters_per_epoch)
            with torch.no_grad():
                ssim_eval, psnr_eval = test(net, loader_test)

            log = f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}'
            print(log)
            with open(os.path.join(opt.saved_data_dir, 'log.txt'), 'a') as f:
                f.write(log + '\n')
            
            
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            psnr_log.append(psnr_eval)
            plot_psnr_log(psnr_log, epoch, opt.saved_plot_dir)

            # 每规定次数保存权重
            # 先保存最优权重
            if psnr_eval > max_psnr:
                max_psnr = max(max_psnr, psnr_eval)
                max_ssim = max(max_ssim, ssim_eval)
                print(
                    f'\n model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pk')
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict(),
                    'optimizer': optim.state_dict()
                }, saved_best_model_path)
            
            #保存权重
            save_single_model_path = os.path.jois(opt.save_mode_dir, str(epoch) + '.pk')
            torch.save({
                'epoch': epoch,
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict(),
                    'optimizer': optim.state_dict()
            }, save_single_model_path)

            loader_train_iter = iter(loader_train) # 取下一次数据集
            # 保存 ssim和 psnr数据
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)
            np.save(os.path.join(opt.saved_data_dir, 'ssims.npy'), ssims)

def pad_img(x, patch_size):
    _,_, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - h % patch_size) % patch_size

    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def test (net, loader_test):
    #开启验证模式
    net.eval()
    torch.cuda.empty_cache() # 手动释放 PyTorch 中未被使用的 GPU 缓存内存
    ssims = []
    psnrs = []

    for i, (inputs, targets, hazy_name) in enumerate(loader_test):
        
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            # 训练的时候是裁切256 * 256 训练，推理的时候要原图直接推，所以要padd一下
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            pred = net(inputs).clamp(0,1)
            # 还原图像尺寸
            pred = pred[:, :, :H, :W]
            # save_path = os.path.join(opt.saved_infer_dir, hazy_name[0])
            # save_image(pred, save_path)
        # 计算 ssim和 psnr
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)




def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)                            # 设置numpy的随机种子， numpy 用于数据预处理（生成随机索引， 随机数据增强）
    np.random.seed(seed) 
    torch.manual_seed(seed)                         # 控制CPU上的随机操作（初始化权重， 随机采样）
    torch.cuda.manual_seed(seed)                    # 控制GPU上的随机种子 
    torch.backends.cudnn.deterministic = True       # 强制CuDNN 使用确定性算法，CuDNN 为了优化加速，部分操作可能会使用非确定性算法，开启选项后会牺牲一点速度保持结果一致

if __name__ == "__main__":
    
    set_seed_torch(666)

    # 读数据集，训练集和测试集
    train_dir = '../dataset/UAV/train'
    test_dir = '../dataset/UAV/test'
    # 实例化数据对象
    train_set = TrainDataset(os.path.join(train_dir, 'hazy'), os.path.join(train_dir, 'clear'))
    test_set = TestDataset(os.path.join(test_dir, 'hazy'), os.path.join(test_dir, 'clear'))

    # 用dataload 加载数据集
    loader_train = DataLoader(dataset=train_set, batch_size=8, shuffle=True, num_workers=12)
    loader_test = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=4)

    # 加载网路
    net = DiGPaRNet(base_dim=32) # 3 -> 32 这是为什么？UNet 都是直接从3->32么？有什么更好的效果？
    net = net.to(opt.device)

    # 计算多少个epoch
    epoch_size = len(loader_train)
    print("epoch_size:", epoch_size)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    # 定义损失函数
    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(ContrastLoss(ablation=False))

    # 定义优化器
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.start_lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad() #梯度清零

    # 训练(把数据，网路，损失函数，优化器)一起放到放入训练流程
    train(net, loader_train, loader_test, optimizer, criterion)