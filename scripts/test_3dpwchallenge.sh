
TEST_CONFIGS='configs/test.yml'

GPUS=$(cat $TEST_CONFIGS | shyaml get-value ARGS.GPUS)
CUDA_VISIBLE_DEVICES=${GPUS} python -u lib/evaluation/collect_3DPW_results.py --GPUS=${GPUS} --configs_yml=${TEST_CONFIGS}