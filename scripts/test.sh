
TEST_CONFIGS='configs/test.yml'

GPUS=$(cat $TEST_CONFIGS | shyaml get-value ARGS.gpu)
CUDA_VISIBLE_DEVICES=${GPUS} python -m romp.test --configs_yml=${TEST_CONFIGS}