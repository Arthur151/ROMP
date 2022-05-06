# kernprof -v -l --persp 
EVAL_MODE=0
TEST_MODE=0
SUBMIT_MODE=0

TRAIN_CONFIGS='configs/v7.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)
#CUDA_VISIBLE_DEVICES=${GPUS} python -u -m torch.distributed.launch --nproc_per_node=4  romp.train --GPUS=${GPUS} --configs_yml=${TRAIN_CONFIGS}
CUDA_VISIBLE_DEVICES=${GPUS} nohup python -u -m torch.distributed.launch --nproc_per_node=4 romp.train --GPUS=${GPUS} --configs_yml=${TRAIN_CONFIGS} --distributed_training=1 > '../log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2>&1 &
