
TEST_MODE=1

TEST_CONFIGS='configs/basic_test.yml'
GPUS=$(cat $TEST_CONFIGS | shyaml get-value ARGS.GPUS)
CUDA_VISIBLE_DEVICES=${GPUS} python core/test.py --gpu=${GPUS} --configs_yml=${TEST_CONFIGS}