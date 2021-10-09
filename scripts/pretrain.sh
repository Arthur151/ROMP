
TRAIN_CONFIGS='configs/pretrain.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

CUDA_VISIBLE_DEVICES=${GPUS} nohup python -u -m romp.pretrain --GPUS=${GPUS} --configs_yml=${TRAIN_CONFIGS} > 'log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2>&1 &