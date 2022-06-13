import os
import argparse, json
import numpy as np


N_EXPERIMENTS = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--eval_dir', default='eval', help='Log dir [default: eval]')
    parser.add_argument('--exp', help='Name of experiment')
    parser.add_argument('--n_objs', help='Max number of objects')
    parser.add_argument('--checkpnt', help='Checkpoint that is evaluated for all experiments (if possible)')
    FLAGS = parser.parse_args()
    return FLAGS


if __name__ == '__main__':

    FLAGS = parse_arguments()

    exp_eval_list = []

    for i in range(N_EXPERIMENTS):

        cur_exp = os.path.join(FLAGS.log_dir, FLAGS.exp.replace('xxx', str(i)))
        exp_dir = os.path.join(cur_exp, FLAGS.eval_dir)
        print(exp_dir)
        if not os.path.exists(exp_dir):
            print('Directory does not exist: ', exp_dir)
            continue

        all_json_paths = []
        json_path = ""
        for f in os.listdir(exp_dir):
            if '.json' in f:
                all_json_paths.append(os.path.join(exp_dir, f))
                if FLAGS.checkpnt in f:
                    json_path = os.path.join(exp_dir, f)
        if json_path == "" and len(all_json_paths)>0:
            json_path = all_json_paths[-1]
        if json_path == "":
            print('No JSON file found in ', exp_dir)
            continue

        with open(os.path.join(json_path)) as json_file:
            eval_stats = json.load(json_file)

            exp_eval_list.append(eval_stats)

    combination = {}
    final = {}
    final_short = {}
    all_exps = range(len(exp_eval_list))
    final['epochs'] = [exp_eval_list[i]['epoch'] for i in all_exps]

    keys_float = [ 'inst_ap', 'inst_ap05', 'inst_ar05', 'inst_f1-05',
                   'rgb_rms', 'rgb_psnr', 'rgb_ssim',
                   'depth_rmse', 'depth_abs_rel_diff', 'depth_squ_rel_diff',
                   'pos_3d', 'extr_r', 'extr_r_sym', 'extr_r_med', 'extr_r_sym_med', 'pos_2d', 'extr_z', 'extr_s']

    for k in keys_float:
        if k not in exp_eval_list[i]:
            continue
        combination[k] = []
        for i in all_exps:
            print(k, i, exp_eval_list[i][k])
            score = exp_eval_list[i][k]
            if '[' in score:
                score = score.split(',')[0][1:]
            combination[k].append(np.float(score))
    for k in keys_float:
        if k not in combination.keys():
            continue
        comb = combination[k]
        final[k] = [['max', np.around(np.nanmax(comb), 3), all_exps[np.nanargmax(comb)]],
                    ['min', np.around(np.nanmin(comb), 3), all_exps[np.nanargmin(comb)]],
                    ['mean', np.around(np.nanmean(comb), 3)],
                    ['std', np.around(np.nanstd(comb), 3)],
                    ['non-nan', np.count_nonzero(~np.isnan(comb))]]
        final_short[k] = np.around(np.mean(comb), 3)

        print(k, final[k])

    path = os.path.join(FLAGS.log_dir, FLAGS.exp + '_' + FLAGS.eval_dir + '_checkpnt' + FLAGS.checkpnt + '.json')
    with open(path, 'w') as f:
        json.dump(final, f, indent=2)
    path = os.path.join(FLAGS.log_dir, FLAGS.exp + '_' + FLAGS.eval_dir + '_checkpnt' + FLAGS.checkpnt + '_short.json')
    with open(path, 'w') as f:
        json.dump(final_short, f, indent=2)


