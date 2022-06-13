import importlib
import psutil
import numpy as np
import pickle
import json


# --------------------------------------------------
# --- General Functions

def load_module_from_log(name, file):
    """
    Load module from backup file (i.e. not in project structure)
    :param name:    string, name of module
    :param file:    string, path to module file
    :return:        imported module
    """
    assert (name in file)
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def get_mem():
    """
    Return used system memory in MB
    """
    output = "Mem=[" + str(int(psutil.virtual_memory().used / 1024 / 1024)) + "/" + str(
        int(psutil.virtual_memory().total / 1024 / 1024)) + " MB]"
    return output


def split_time_delta(time_delta):
    time_delta_hours = time_delta.seconds // 3600
    time_delta_minutes = (time_delta.seconds - time_delta_hours * 3600) // 60
    time_delta_seconds = (time_delta.seconds - time_delta_hours * 3600 - time_delta_minutes * 60)

    return [time_delta_hours, time_delta_minutes, time_delta_seconds]


# --------------------------------------------------
# --- Evaluation Functions

# -- Instance

def inst_ap(gt_inst, pred_inst, th=0.5, min_n_pxls=1):
    """
    Instance segmantation evaluation measures
    :param gt_inst:     (S, H, W, 1)
    :param pred_inst:   (S, H, W, 1)
    :param th:          int
    :return:            [float, float, float, (int, int), int]: prec_val, rec_val, f1, match_id, n_found_obj
    """
    assert gt_inst.shape[0] == pred_inst.shape[0]
    assert min_n_pxls > 0
    if gt_inst.shape[-1] == 1:
        gt_inst = np.squeeze(gt_inst, axis=-1)
    if pred_inst.shape[-1] == 1:
        pred_inst = np.squeeze(pred_inst, axis=-1)

    val_gt = np.sum(gt_inst, axis=(1, 2))
    val_gt = (val_gt >= min_n_pxls).astype(int)

    val_pred = np.sum(pred_inst, axis=(1, 2))
    val_pred = (val_pred >= min_n_pxls).astype(int)

    tp = np.zeros((pred_inst.shape[0]))
    fp = np.ones((pred_inst.shape[0]))
    fn = np.ones((gt_inst.shape[0]))

    match_id = []
    n_found_obj = []

    for n in range(pred_inst.shape[0]):
        mask_pred = pred_inst[n]
        for m in range(gt_inst.shape[0]):
            if tp[n] == 1 or fn[m] == 0:  # already found/ used
                continue
            mask_gt = gt_inst[m]
            intersect = mask_pred * mask_gt
            intersect = np.sum(intersect)
            union = np.maximum(mask_pred, mask_gt)
            union = np.sum(union)
            assert (union >= intersect)
            with np.errstate(divide='ignore', invalid='ignore'):
                iou = np.nan_to_num(intersect / union)
            if iou >= th:
                tp[n] = 1.
                fp[n] = 0.
                fn[m] = 0.
                match_id.append([n, m])

    tp_val = np.sum(val_pred * tp)
    fp_val = np.sum(val_pred * fp)
    fn_val = np.sum(val_gt * fn)

    n_found_obj.append(tp_val)

    prec_val = tp_val / (tp_val + fp_val)
    rec_val = tp_val / (tp_val + fn_val)

    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.nan_to_num(2 * (prec_val * rec_val) / (prec_val + rec_val))

    prec_val = np.nan_to_num(prec_val)
    rec_val = np.nan_to_num(rec_val)

    return prec_val, rec_val, f1, match_id, n_found_obj


# --------------------------------------------------
# --- Matrix Transformation Functions for Rendering

def get_rotation_mat(alpha, axis):
    if axis == 'x':
        return [[1., 0., 0.], [0., np.cos(alpha), -np.sin(alpha)], [0., np.sin(alpha), np.cos(alpha)]]
    elif axis == 'y':
        return [[np.cos(alpha), 0., np.sin(alpha)], [0., 1., 0.], [-np.sin(alpha), 0., np.cos(alpha)]]
    elif axis == 'z':
        return [[np.cos(alpha), -np.sin(alpha), 0.], [np.sin(alpha), np.cos(alpha), 0.], [0., 0., 1.]]
    else:
        return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


def get_rotation_mat_batch(alpha_batch, axis):
    mats = []
    for alpha in alpha_batch:
        mats.append(get_rotation_mat(alpha, axis))

    mats = np.array(mats)

    return mats


def get_rotation_matrix(euler):
    Rx = get_rotation_mat(euler[0], 'x')
    Ry = get_rotation_mat(euler[1], 'y')
    Rz = get_rotation_mat(euler[2], 'z')

    R = np.matmul(Rz, np.matmul(Ry, Rx))

    return R


# adapted from https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
def get_blender_cam(R, t):
    R_bcam2cv = np.array(
        ((1, 0, 0),
         (0, -1, 0),
         (0, 0, -1)))

    R_world2bcam = np.transpose(R)

    # Convert camera location to translation vector used in coordinate changes
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * np.matmul(R_world2bcam, np.expand_dims(t, -1))

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = np.matmul(R_bcam2cv, R_world2bcam)
    T_world2cv = np.matmul(R_bcam2cv, T_world2bcam)

    T_world2cv = np.squeeze(T_world2cv)

    return R_world2cv, T_world2cv


# --------------------------------------------------
# ---
def save_config(cnfg, path):
    with open(path, 'wb') as fp:
        pickle.dump(cnfg, fp)

    # readable version
    def dict_serializable(d):
        tmp = {}
        for k, v in d.items():
            if isinstance(v, dict):
                tmp[k] = dict_serializable(v)
            else:
                if isinstance(v, np.ndarray):
                    tmp[k] = v.tolist()
                else:
                    tmp[k] = v
        return tmp
    dict_tmp = dict_serializable(cnfg)
    json.dump(dict_tmp, open(path.replace('.pkl', '.json'), 'w'), indent=4)


def load_config(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


class LogFile:

    def __init__(self, log_path, new=False):
        self.log_path = log_path
        if new:
            self.log_fout = open(log_path, 'w')
        else:
            self.log_fout = open(log_path, 'a+')

    def write(self, out_str):
        self.log_fout.write(out_str + '\n')
        self.log_fout.flush()
        print(out_str)

    def read_lines(self):
        fp = open(self.log_path, 'r')
        lines = fp.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace('\n', '').split(' ')
        return lines

