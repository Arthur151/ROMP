
DEMO_SINGLE_MODE=1
DEMO_VIDEO_MODE=0
WEBCAM_MODE=0
EVALUATION_MODE=0

DEMO_SINGLE_CONFIGS='configs/single_image.yml'
DEMO_VIDEO_CONFIGS='configs/video.yml'
WEBCAM_CONFIGS='configs/webcam.yml'
EVALUATION_CONFIGS='configs/eval_3dpw.yml'

if [ "$DEMO_SINGLE_MODE" = 1 ]
then
    GPUS=$(cat $DEMO_SINGLE_CONFIGS | shyaml get-value ARGS.GPUS)
    CUDA_VISIBLE_DEVICES=${GPUS} python core/test.py --gpu=${GPUS} --configs_yml=${DEMO_SINGLE_CONFIGS}
elif [ "$DEMO_VIDEO_MODE" = 1 ]
then
    GPUS=$(cat $DEMO_VIDEO_CONFIGS | shyaml get-value ARGS.GPUS)
    CUDA_VISIBLE_DEVICES=${GPUS} python -u core/test.py --gpu=${GPUS} --configs_yml=${DEMO_VIDEO_CONFIGS}
elif [ "$WEBCAM_MODE" = 1 ]
then
    GPUS=$(cat $WEBCAM_CONFIGS | shyaml get-value ARGS.GPUS)
    CUDA_VISIBLE_DEVICES=${GPUS} python -u core/test.py --gpu=${GPUS} --configs_yml=${WEBCAM_CONFIGS}
elif [ "$EVALUATION_MODE" = 1 ]
then
    GPUS=$(cat $EVALUATION_CONFIGS | shyaml get-value ARGS.GPUS)
    CUDA_VISIBLE_DEVICES=${GPUS} python -u core/benchmarks_evaluation.py --gpu=${GPUS} --configs_yml=${EVALUATION_CONFIGS}
fi    