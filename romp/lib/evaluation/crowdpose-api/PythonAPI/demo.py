from crowdposetools.coco import COCO
from crowdposetools.cocoeval import COCOeval

set_name = 'val' #'test' # 
gt_file = '../annotations/crowdpose_{}.json'.format(set_name)
preds = '../annotations/V1_crowdpose_{}_preds_CAR0.2_ct0.2_mp64.json'.format(set_name)

cocoGt = COCO(gt_file)
cocoDt = cocoGt.loadRes(preds)
cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
