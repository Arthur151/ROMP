TRAIN_CONFIGS='configs/trace.yml'

GPUS=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.GPUS)
DATASET=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.datasets)
TAB=$(cat $TRAIN_CONFIGS | shyaml get-value ARGS.tab)

CUDA_VISIBLE_DEVICES=${GPUS} python -u -m train_video --gpu=${GPUS} --configs_yml=${TRAIN_CONFIGS}
#CUDA_VISIBLE_DEVICES=${GPUS} nohup python -u -m train_video --gpu=${GPUS} --configs_yml=${TRAIN_CONFIGS} > '../project_data/trace_data/log/'${TAB}'_'${DATASET}'_g'${GPUS}.log 2>&1 &
