# Data

The VidSTG dataset can download from https://github.com/Guaranteer/VidSTG-Dataset

# Training
```python
python -m torch.distributed.launch --nproc_per_node=8 --user_env main.py --load .../pretrained_resnet101_checkpoint.pth --ema --no_contrastive_align_loss   
```
