import json
import numpy as np

path_prefix = './log/clevr_mosnet-org(train_mosnet_optimized_sil_400_ep'
opt_method = '_eval_w_r_and_c'
path_suffix = opt_method + ')/eval/val_ep400.json'
voxel_reconstruction_path_suffix = opt_method + ')/eval/val_ep400_3d_reconstruction.json'
voxel_reconstruction_single_objs_path_suffix = opt_method + ')/eval/val_ep400_3d_reconstruction_single_objs.json'

combined_dict = {}
output_dict = {}

with open(path_prefix + path_suffix) as json_file:
    input_dict = json.load(json_file)
for k, v in input_dict.items():
    combined_dict[k] = []
combined_dict['voxel_eval_mean_iou'] = []
combined_dict['voxel_eval_mean_iou_single_objs'] = []
combined_dict['voxel_eval_num_objs_detected_single_objs'] = []
combined_dict['voxel_eval_iou_over_thresh'] = []
combined_dict['voxel_eval_iou_over_thresh_single_objs'] = []

for i in range(5):
    if i==0:
        with open(path_prefix + path_suffix) as json_file:
            input_dict = json.load(json_file)
        with open(path_prefix + voxel_reconstruction_path_suffix) as json_file:
            voxel_reconstruction_dict = json.load(json_file)
        with open(path_prefix + voxel_reconstruction_single_objs_path_suffix) as json_file:
            voxel_reconstruction_single_objs_dict = json.load(json_file)
    else:
        with open(path_prefix + '_rep_' + str(i) + path_suffix) as json_file:
            input_dict = json.load(json_file)
        with open(path_prefix + '_rep_' + str(i) + voxel_reconstruction_path_suffix) as json_file:
            voxel_reconstruction_dict = json.load(json_file)
        with open(path_prefix + '_rep_' + str(i) + voxel_reconstruction_single_objs_path_suffix) as json_file:
            voxel_reconstruction_single_objs_dict = json.load(json_file)
    for k, v in input_dict.items():
        combined_dict[k].append(v)
    combined_dict['voxel_eval_mean_iou'].append(voxel_reconstruction_dict['mean_iou'])
    combined_dict['voxel_eval_mean_iou_single_objs'].append(voxel_reconstruction_single_objs_dict['mean_iou'])
    combined_dict['voxel_eval_num_objs_detected_single_objs'].append(voxel_reconstruction_single_objs_dict['num_objs_detected'])

    iou_vals = []
    for i in voxel_reconstruction_dict['all_iou_vals']:
        iou_vals.append(float(i))
    thresh = 0.7
    num_greater_than_thresh = np.sum((np.array(iou_vals)>thresh).astype(int))
    combined_dict['voxel_eval_iou_over_thresh'].append(num_greater_than_thresh)

    iou_vals_single_objs = []
    for i in voxel_reconstruction_single_objs_dict['all_iou_vals']:
        iou_vals_single_objs.append(float(i))
    thresh = 0.85
    num_greater_than_thresh = np.sum((np.array(iou_vals_single_objs)>thresh).astype(int))
    combined_dict['voxel_eval_iou_over_thresh_single_objs'].append(num_greater_than_thresh)


for k, v in combined_dict.items():
    if k.startswith('inst') or k.startswith('rgb') or k.startswith('depth') or k.startswith('pos_3d') or k.startswith('extr_r') or k.startswith('voxel'):
        values_list = []
        for i in v:
            values_list.append(float(i))
        output_dict[k + '_mean_std'] = [np.mean(values_list), np.std(values_list)]
    #if k.startswith('n_found_obj') or k.startswith('extr_x') or k.startswith('extr_y') or k.startswith('extr_z') or k.startswith('extr_s'):
    #    output_dict[k] = output_dict[k] + v

# for k, v in combined_dict.items():
#     print("{}: {}".format(k, v))

print('\n' + path_prefix+path_suffix)
print("\noutput_dict:")
for k, v in output_dict.items():
    print("{}: {}".format(k, v))








# path = './log/clevr_mosnet-org(train_mosnet_optimized_sil_normal_lin_weight_w_and_wo_sil_mask_intersection_loss_lin_weight_800_ep_eval_w_r_and_c_anti_aliasing_depth_w_aa)/eval/'
# print(path)
#
# with open(path + 'val_ep800_first_half.json') as json_file:
#     first_half_dict = json.load(json_file)
#
# with open(path + 'val_ep800_second_half.json') as json_file:
#     second_half_dict = json.load(json_file)
#
#
# print(first_half_dict)
# print('\n')
# print(second_half_dict)
# print('\n')
#
# combined_dict = {}
# for k, v in first_half_dict.items():
#     combined_dict[k] = v
#
# for k, v in second_half_dict.items():
#     if k.startswith('inst') or k.startswith('rgb') or k.startswith('depth') or k.startswith('pos_3d') or k.startswith('extr_r'):
#         val_from_first_half = float(combined_dict[k])
#         val_from_second_half = float(v)
#         combined_dict[k] = str((val_from_first_half + val_from_second_half) / 2)
#     if k.startswith('n_found_obj') or k.startswith('extr_x') or k.startswith('extr_y') or k.startswith('extr_z') or k.startswith('extr_s'):
#         combined_dict[k] = combined_dict[k] + v
#
# print(combined_dict)
# print('\n')
#
# with open(path + 'val_ep800.json', 'w') as f:
#     json.dump(combined_dict, f, indent=2)
