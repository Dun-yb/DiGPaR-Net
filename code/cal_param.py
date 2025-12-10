from model import DiGPaRNet
import torch
from thop import profile, clever_format
from torchstat import stat

if __name__ =='__main__':
    # model = torchvision.models.AlexNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(1, 3, 256, 256).to(device)

    model = DiGPaRNet().to(device)
    
    flops, params = profile(model, inputs=(x, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")

   
    
    print(f"Params: {params}")
    print(f"FLOPs: {flops}")
