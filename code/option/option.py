import os, argparse
import json

# 推理的参数设置 option
parser = argparse.ArgumentParser()


# 权重位置
# 数据集位置
# 需要保存图片的输出位置
parser.add_argument('--exp_dir', type=str, default='../experiment')
parser.add_argument('--dataset', type=str, default='UAV')
parser.add_argument('--model_name', type=str, default='DiGPaRNet', help='experiment name')
parser.add_argument('--saved_infer_dir', type=str, default='saved_infer_dir')
parser.add_argument('--saved_heatmap_results', type=str, default='saved_heatmap_dir')


parser.add_argument('--pre_trained_model', type=str, default='null', help='path of pre trained model for resume training')
parser.add_argument('--save_infer_results', action='store_true', default=False, help='save the infer results during validation')
parser.add_argument('--save_heatmap_results', action='store_true', default=False, help='save the infer heatmap results during validation')

opt = parser.parse_args()

opt.val_dataset_dir = os.path.join('../dataset/', opt.dataset, 'test')
exp_dataset_dir = os.path.join(opt.exp_dir, opt.dataset)

exp_model_dir = os.path.join(exp_dataset_dir, opt.model_name)


if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)
if not os.path.exists(exp_dataset_dir):
    os.mkdir(exp_dataset_dir)

opt.saved_infer_dir = os.path.join(exp_model_dir, opt.pre_trained_model.split('.pth')[0])

opt.saved_infer_dir = os.path.join(exp_model_dir, opt.pre_trained_model.split('.pth')[0])
if not os.path.exists(exp_model_dir):
    os.mkdir(exp_model_dir)
    os.mkdir(opt.saved_infer_dir)
if not os.path.exists(opt.saved_infer_dir):
    os.mkdir(opt.saved_infer_dir)

with open(os.path.join(exp_model_dir, 'args.txt'), 'w') as f:
    json.dump(opt.__dict__, f, indent=2)