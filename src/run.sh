
TEST_MODE=1
WEBCAM_MODE=0

TEST_CONFIGS='configs/basic_test.yml'
WEBCAM_CONFIGS='configs/basic_webcam.yml'

if [ "$TEST_MODE" = 1 ]
then
    GPUS=$(cat $TEST_CONFIGS | shyaml get-value ARGS.GPUS)
    CUDA_VISIBLE_DEVICES=${GPUS} python core/test.py --gpu=${GPUS} --configs_yml=${TEST_CONFIGS}
elif [ "$WEBCAM_MODE" = 1 ]
then
    GPUS=$(cat $WEBCAM_CONFIGS | shyaml get-value ARGS.GPUS)
    CUDA_VISIBLE_DEVICES=${GPUS} python -u core/test.py --gpu=${GPUS} --configs_yml=${WEBCAM_CONFIGS}
fi    