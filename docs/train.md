## Train

1. Please first prepare your datasets follow [dataset.md](dataset.md) and finish the installation follow [installation.md](installation.md).

2. Run the script:
You can train ROMP via
```
# go into the path/to/ROMP
cd ROMP
# To train the ROMP with HRNet-32 as backbone, please run
sh scripts/V1_train.sh
# To train the ROMP with ResNet-50 as backbone, please run
sh scripts/V1_train_resnet.sh
```
To re-implement the results in Tab.3 of our paper, you can fine-tune the trained model on the 3DPW dataset via
```
# To fine-tune the ROMP with HRNet-32 as backbone, please run
sh scripts/V1_hrnet_3dpwft.sh
# To fine-tune the ROMP with ResNet-50 as backbone, please run
sh scripts/V1_resnet_3dpwft.sh
```
You can change the configurations (e.g. used GPUs, batch size) in the corresponding configs (.yml) in ROMP/configs.
