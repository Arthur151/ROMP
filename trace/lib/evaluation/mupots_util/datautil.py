import scipy.io as sio 

def load_annot(fname):
    def parse_pose(dt):
        res = {}
        annot2 = dt['annot2'][0,0]
        annot3 = dt['annot3'][0,0]
        annot3_univ = dt['univ_annot3'][0,0]
        is_valid = dt['isValidFrame'][0,0][0,0]
        res['annot2'] = annot2
        res['annot3'] = annot3
        res['annot3_univ'] = annot3_univ
        res['is_valid'] = is_valid
        return res 
    data = sio.loadmat(fname)['annotations']
    results = []
    num_frames, num_inst = data.shape[0], data.shape[1]
    for j in range(num_inst):
        buff = []
        for i in range(num_frames):
            buff.append(parse_pose(data[i,j]))
        results.append(buff)
    return results
    
def load_occ(fname):
    data = sio.loadmat(fname)['occlusion_labels']
    results = []
    num_frames, num_inst = data.shape[0], data.shape[1]
    for i in range(num_frames):
        buff = []
        for j in range(num_inst):
            buff.append(data[i][j])
        results.append(buff)
    return results
