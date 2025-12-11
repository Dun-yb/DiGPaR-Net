# DiGPaR-Net

## Preparation

### Download

**Dataset:**

* Download UAV-Rain1k from [here](https://github.com/cschenxiang/UAV-Rain1k).
* Download HazyDet from [here](https://github.com/GrokCV/HazyDet).
* Download SateHaze1k from [here](https://www.dropbox.com/s/k2i3p7puuwl2g59/SateHaze1k.zip?dl=0).
* Download RICE (RICE1 and RICE2) from [here](https://github.com/BUPTLdy/RICE_DATASET).

**Code:**

* The implementation code will be released upon the acceptance of the manuscript.

The final file path should be the same as the following:

```
┬─ datasets
│   ├─ UAV
│   │   ├─train
│   │   │   ├─clear
│   │   │   └─hazy
│   │   └─test
│   │       ├─clear
│   │       └─hazy
│   └─ ...
├─ code
│   └─ ...
└─ trained_models
    └─ ...
```

### Install

We conduct all the experiments on Python 3.8 + PyTorch 2.4.1 + CUDA 12.1

1. Clone our repository
```
git clone https://github.com/Dun-yb/DiGPaR-Net.git
cd DiGPaR-Net
```

2. Make conda environment
```
conda create -n digparnet python=3.8
conda activate digparnet
```

3. Install dependencies
```
conda install cudatoolkit
pip install -r requirements.txt
```

## Training and Testing

### Train
```
CUDA_VISIBLE_DEVICES=X python train.py --epochs 200 --iters_per_epoch 5000 --finer_eval_step 900000 --w_loss_L1 1.0 --w_loss_CR 0.1 --start_lr 0.0001 --end_lr 0.000001 --exp_dir ../experiment/ --model_name DiGPaR-Net --dataset UAV
```

### Test
```
CUDA_VISIBLE_DEVICES=X python eval.py --dataset UAV --model_name xxxx --pre_trained_model xxxxx.pk --save_infer_results
```

## Citation
```
@ARTICLE{,
  author={},
  journal={}, 
  title={}, 
  year={},
  volume={},
  number={},
  pages={},
  keywords={},
  doi={}}
```