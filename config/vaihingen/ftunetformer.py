from torch.utils.data import DataLoader
from geoseg.losses import *
from geoseg.datasets.vaihingen_dataset import *
from catalyst.contrib.nn import Lookahead
from catalyst import utils
from geoseg.models.FTUNetFormer import ft_unetformer_tiny

# training hparam
# max_epoch = 45
max_epoch = 225
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 16
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1
num_classes = len(CLASSES)
classes = CLASSES
test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "ftunetformer_tiny_vaihingen_225"
weights_path = "model_weights/vaihingen/{}".format(weights_name)
#test_weights_name = "ftunetformer_tiny_vaihingen_225-v2"
# test_weights_name = "ftunetformer_tiny_vaihingen_225-v1"
test_weights_name = "ftunetformer_tiny_vaihingen_225"
# test_weights_name = "last"
log_name = 'vaihingen/{}'.format(weights_name)
monitor = 'val_F1'
monitor_mode = 'max'
save_top_k = 3
save_last = False
check_val_every_n_epoch = 5
gpus = [0]
# gpus = 0

strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None
# 2023.1.10新增控制预训练权重参数
backbone_model_pretrain_path = None
# resume_ckpt_path = 'model_weights/potsdam/unetformer-r18-768crop-ms-e45/unetformer-r18-768crop-ms-e45.ckpt'
#  define the network
net = ft_unetformer_tiny(pretrained=True, num_classes=num_classes, freeze_stages=-1, decoder_channels=96,
                  weight_path='pretrain_weights/stseg_tiny.pth')



# define the loss
loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False

train_dataset = VaihingenDataset(data_root='data/vaihingen/train', mode='train',
                                 mosaic_ratio=0.25, transform=train_aug)

val_dataset = VaihingenDataset(transform=val_aug)
test_dataset = VaihingenDataset(data_root='data/vaihingen/test',
                                transform=val_aug)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

