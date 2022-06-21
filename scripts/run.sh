
IMAGE_MODE=1
VIDEO_MODE=0
WEBCAM_MODE=0
EVALUATION_MODE=0

IMAGE_CONFIGS='configs/image.yml'
VIDEO_CONFIGS='configs/video.yml'
WEBCAM_CONFIGS='configs/webcam.yml'
EVALUATION_CONFIGS='configs/eval_3dpw.yml'

if [ "$IMAGE_MODE" = 1 ]
then
    GPUS=$(cat $IMAGE_CONFIGS | shyaml get-value ARGS.gpu)
    CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.predict.image --configs_yml=${IMAGE_CONFIGS}
elif [ "$VIDEO_MODE" = 1 ]
then
    GPUS=$(cat $VIDEO_CONFIGS | shyaml get-value ARGS.gpu)
    CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.predict.video --configs_yml=${VIDEO_CONFIGS}
elif [ "$WEBCAM_MODE" = 1 ]
then
    GPUS=$(cat $WEBCAM_CONFIGS | shyaml get-value ARGS.gpu)
    CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.predict.webcam --configs_yml=${WEBCAM_CONFIGS}
elif [ "$EVALUATION_MODE" = 1 ]
then
    GPUS=$(cat $EVALUATION_CONFIGS | shyaml get-value ARGS.gpu)
    CUDA_VISIBLE_DEVICES=${GPUS} python -u -m romp.benchmarks_evaluation --configs_yml=${EVALUATION_CONFIGS}
fi    