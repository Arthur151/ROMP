# kernprof -v -l --persp 
EVAL_MODE=0
TEST_MODE=0
SUBMIT_MODE=0

TRAIN_CONFIGS='configs/v7.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.dataset)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)
#CUDA_VISIBLE_DEVICES=${GPUS} torchrun --nproc_per_node=4 -m romp.train --GPUS=${GPUS} --configs_yml=${TRAIN_CONFIGS} --distributed_training=1
CUDA_VISIBLE_DEVICES=${GPUS} nohup torchrun --nproc_per_node=4 -m romp.train --GPUS=${GPUS} --configs_yml=${TRAIN_CONFIGS} --distributed_training=1 > '../log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2>&1 &
