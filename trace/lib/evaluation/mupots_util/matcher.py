import numpy as np 
import lap

def norm_by_bone_length(pred, gt, o1, trav):
    mapped_pose = pred.copy()

    for i in range(len(trav)):
        idx = trav[i]
        gt_len = np.linalg.norm(gt[:,idx] - gt[:,o1[i]])
        pred_vec = pred[:, idx] - pred[:,o1[i]]
        pred_len = np.linalg.norm(pred_vec)
        mapped_pose[:, idx] = mapped_pose[:, o1[i]] + pred_vec * gt_len / pred_len
    return mapped_pose

def procrustes(predicted, target):
    predicted = predicted.T 
    target = target.T
    predicted = predicted[None, ...]
    target = target[None, ...]

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    return predicted_aligned[0].T

def match_3d(pose1, pose2, o1=None, threshold=250):
    # pose: [2, num_pts]
    # vis: [1, num_pts]
    matches = []
    p2 = np.float32(pose2)
    if o1 is not None:
        p2 = p2 - p2[:,:,14:15]
    for i in range(len(pose1)):
        p1 = np.float32(pose1[i])
        p1 = p1 - p1[:,14:15]
        diffs = []
        for j in range(len(p2)):
            p = p2[j]
            p = procrustes(p, p1)
            diff = np.sqrt(np.power(p-p1,2).sum(axis=0)).mean()
            diffs.append(diff)
        diffs = np.float32(diffs)
        idx = np.argmin(diffs)
        if diffs.min()>threshold:
            matches.append(-1)
        else:
            matches.append(idx)
    return matches

def match_2d(kp2d_gts, kp2d_preds, thresh=100):
    # kp2d_gts, shape N1 x 2 x K, N1 is the number of gt kp2d
    # kp2d_preds, shape N2 x 2 x K, N2 is the number of pred kp2d
    matched_ids = np.ones(len(kp2d_gts), dtype=np.int32) * -1
    kp2d_gts = np.array(kp2d_gts)
    cost_matrix = np.ones((len(kp2d_preds), len(kp2d_gts)))
    # calc the distance matrix between each pair of the gt and the pred
    # each row in dist. mat. represent the distance between a predict kp2d and all gts
    for pid, kp2d_p in enumerate(kp2d_preds):
        dist = np.linalg.norm(kp2d_gts-kp2d_p[None], axis=1, ord=2).mean(-1)
        cost_matrix[pid] = dist
    #print(cost_matrix)
    cost, matched_gtsID_for_each_preds, matched_predsID_for_each_gts = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh) #
    for pred_id, gt_id in enumerate(matched_gtsID_for_each_preds):
        if gt_id != -1:
            matched_ids[gt_id] = pred_id
    # print(matched_gtsID_for_each_preds, matched_predsID_for_each_gts)
    # print(matched_ids)
    return matched_ids

