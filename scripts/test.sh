
TEST_CONFIGS='configs/test.yml'

GPUS=$(cat $TEST_CONFIGS | shyaml get-value ARGS.GPUS)
CUDA_VISIBLE_DEVICES=${GPUS} python -m romp.test --gpu=${GPUS} --configs_yml=${TEST_CONFIGS}