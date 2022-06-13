import os
import argparse
import ntpath
import socket
import itertools
from datetime import datetime
from shutil import copyfile
from termcolor import colored
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# Disable TF info messages (needs to be done before tf import)
#   '0' = all messages are logged (default behavior)
#   '1' = INFO messages are not printed
#   '2' = INFO and WARNING messages are not printed
#   '3' = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from utils import data_provider
from utils.shared_funcs import *
from utils.tf_funcs import *
from utils import tb_funcs

# GPU should not be allocated entirely at beginning
# -- https://stackoverflow.com/questions/34199233/how-to-prevent-tensorflow-from-allocating-the-totality-of-a-gpu-memory
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clevr', help='Data Set')
    parser.add_argument('--data_dir', help='Data dir')
    parser.add_argument('--deepsdf_dir', help='path to pre-trained DeepSDF Model basedir')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--model', default='mosnet-org', help='Model name [mosnet-<variant>]')
    parser.add_argument('--config', default='', help='Configuration file [cnfg_<model>_obj<#>_<dataset>]')
    parser.add_argument('--message', default='', help='Message that specifies settings etc. (for log dir name)')
    parser.add_argument('--no_gpu', default='0', help='Do not use GPU')
    return parser.parse_args()


# --------------------------------------------------
# --- Main Training

def main():
    time_start = datetime.now()
    time_start_str = '{:%d.%m_%H:%M}'.format(datetime.now())

    tf.random.set_seed(42)

    # Create dataset iterator for training and validation
    dataset_train, iterator_train = get_data_iterator(DATA_TRAIN, CONFIG['training'], shuffle=True)
    dataset_val, iterator_val = get_data_iterator(DATA_VAL, CONFIG['training'])
    iterators = {
        'train': iterator_train,
        'val': iterator_val
    }

    print("+++++++++++++++++++++++++++++++")

    # Training parameters
    params = {}
    weights = CONFIG['model'][model_base_name]['l-weights']
    for k, w in weights.items():
        params['w-' + k] = w
    params['gauss_sigma'] = CONFIG['model'][model_base_name]['gauss_sigma']

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=CONFIG['training']['learning_rate'])

    # Operator to save and restore all the variables.
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=MODEL)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)

    # Writer for summary
    writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, 'summary'))

    # Init variables
    ckpt.restore(manager.latest_checkpoint).expect_partial()
    if manager.latest_checkpoint:
        ckpt.step.assign_add(1)
        start_epoch = int(ckpt.step)
        print("Restored from {}, epoch {}".format(manager.latest_checkpoint, start_epoch-1))
    else:
        print('Start new training.')
        start_epoch = 1
        ckpt_deepsdf = tf.train.Checkpoint(step=tf.Variable(1), net=MODEL.decoder_sdf)
        manager_deepsdf = tf.train.CheckpointManager(ckpt_deepsdf, os.path.join(DEEPSDF_DIR, 'ckpts'), max_to_keep=3)
        ckpt_deepsdf.restore(manager_deepsdf.latest_checkpoint).expect_partial()
        print("Load pre-trained DeepSDF model from {}, epoch {}".format(manager_deepsdf.latest_checkpoint,
                                                                        int(ckpt_deepsdf.step)))

    # Create ops dictionary
    ops = {
        'params': params,
        'iterators': iterators,
        'optimizer': opt
    }

    # Iterative Training
    max_epoch = CONFIG['training']['max_epoch']

    for epoch in range(start_epoch, max_epoch+1):
        LOG_FILE.write('----')
        time_now_str = '{:%d.%m_%H:%M}'.format(datetime.now())
        LOG_FILE.write('**** EPOCH %03d - start: %s - now: %s ****' % (epoch, time_start_str, time_now_str))

        if epoch % CONFIG['training']['save_epoch'] == 0:
            run_one_epoch(epoch, MODEL, ops, True, writer)
            run_one_epoch(epoch, MODEL, ops, False, writer)
        else:
            run_one_epoch(epoch, MODEL, ops, True, writer)

        if int(ckpt.step) % CONFIG['training']['save_epoch'] == 0 or epoch == max_epoch:
            save_path = manager.save()
            LOG_FILE.write("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        ckpt.step.assign_add(1)

        # For cluster: break training after some time
        time_delta = datetime.now()-time_start
        time_delta_hours, time_delta_minutes, time_delta_seconds = split_time_delta(time_delta)
        print('Runtime: {} day(s), {} hour(s), {} minute(s), {} seconds'.format(time_delta.days, time_delta_hours,
                                                                                time_delta_minutes, time_delta_seconds))
        # if not socket.gethostname() == 'eiturtindur' and (time_delta_hours >= 1 or (epoch-start_epoch) > 100):
        #     save_path = manager.save()
        #     LOG_FILE.write("Break training at epoch {}, after running time {}; model saved in file: {}.".format(
        #         epoch, str(time_delta), save_path))
        #     exit(3)


def train_step(net, data, optimizer, params, is_training):
    """Trains 'net' on 'example' using 'optimizer' wrt 'params'."""
    with tf.GradientTape() as tape:
        output = net([data['rgb_in']])
        losses = net.get_loss(output, data, params)

    #is_training = False #TODO just to get the ctrl losses

    if is_training:
        variables = net.trainable_variables
        gradients = tape.gradient(losses['loss_total'], variables)
        optimizer.apply_gradients(zip(gradients, variables))
    return output, losses

#TODO
def train_step_with_gt_extrinsics(net, data, optimizer, params, is_training):

    permutations = list(itertools.permutations([0, 1, 2]))
    lowest_loss = tf.Variable(np.inf, dtype=tf.float32)
    for permutation in permutations:
        output_perm = net.call_with_gt_extr([data['rgb_in']], list(permutation))

        losses_perm = net.get_loss(output_perm, data, params)

        if losses_perm['loss_total'] < lowest_loss:
            losses = losses_perm
            lowest_loss = losses_perm['loss_total']


    with tf.GradientTape() as tape:
        output = net([data['rgb_in']])

        #TODO output with the correct extrinsics for each slot

        losses = net.get_loss(output, data, params)

    if is_training:
        variables = net.trainable_variables
        gradients = tape.gradient(losses['loss_total'], variables)
        optimizer.apply_gradients(zip(gradients, variables))
    return output, losses


def run_one_epoch(epoch, net, ops, is_training, summary_writer=None):
    LOG_FILE.write('----')

    if is_training:
        data = DATA_TRAIN
        iterator = ops['iterators']['train']
    else:
        data = DATA_VAL
        iterator = ops['iterators']['val']
        LOG_FILE.write('EVAL')


    num_batches = (data.get_size() // CONFIG['training']['batch_size']) + 1
    b = 10 if (num_batches < 150) else 100

    params = {}
    for k, w in ops['params'].items():
        if isinstance(w, list) and w[0] == 'lin' and len(w)==4:
            params[k] = get_mon_weight(epoch-1, w[1], w[2], w[3])
        elif isinstance(w, list) and w[0] == 'lin' and len(w)==5:
            if epoch <= w[3]: params[k] = w[1]
            elif epoch > w[3] and epoch <= w[4]:
                params[k] = get_mon_weight(epoch-1-w[3], w[1], w[2], w[4]-w[3])
            else: params[k] = w[2]
        elif isinstance(w, float):
            params[k] = w
        else:
            print(colored('[ERROR] train.py, unknown type for loss weight: '+str(k)+str(w), 'red'))
            exit(1)

    loss_dict = {}
    eval_dict = {k: [] for k in AVG_LOSS_LABELS}

    sum_normal_losses = 0.
    sum_normal_losses_l1 = 0.
    sum_depth_losses = 0.
    sum_rgb_losses = 0.
    sum_intersect_losses = 0.

    for batch_id in range(num_batches):

        if batch_id % b == 0:
            print('Current batch/total batch num: %d/%d' % (batch_id, num_batches))

        input_batch = next(iterator)
        input_batch = net.get_input(input_batch)

        output, losses = train_step(net, input_batch, ops['optimizer'], params, is_training)
        # output, losses = train_step_with_gt_extrinsics(net, input_batch, ops['optimizer'], params, is_training)


        sum_normal_losses += losses['l-ctrl_normal']
        sum_normal_losses_l1 += losses['l-ctrl_normal_l1']
        sum_depth_losses += losses['l-ctrl_depth_l1']
        sum_rgb_losses += losses['l-ctrl_rgb_l2']
        sum_intersect_losses += losses['l-ctrl_intersect']
        if batch_id == num_batches-1: #True:
            print("batch {}: normal_l2={}, normal_l1={}, depth_l1={}, rgb={}, intersect={}".format(batch_id, sum_normal_losses/(batch_id+1), sum_normal_losses_l1/(batch_id+1), sum_depth_losses/(batch_id+1), sum_rgb_losses/(batch_id+1), sum_intersect_losses/(batch_id+1)))
            LOG_FILE.write("batch {}: normal_l2={}, normal_l1={}, depth_l1={}, rgb={}, intersect={}".format(batch_id, sum_normal_losses/(batch_id+1), sum_normal_losses_l1/(batch_id+1), sum_depth_losses/(batch_id+1), sum_rgb_losses/(batch_id+1), sum_intersect_losses/(batch_id+1)))

        if not is_training:
            for b, n in itertools.product(range(CONFIG['training']['batch_size']), range(CONFIG['data']['n_images'])):
                def nd_array(a):
                    return tf.make_ndarray(tf.make_tensor_proto(a))

                # --- RGB Image
                img_gt = nd_array(input_batch['rgb_gt'][b, n])  # (H, W, 3), only one image at the moment
                img_pred = nd_array(output['rgb_pred'][b, n])

                eval_dict['rec_rmse'].append(np.sqrt(np.mean(np.square(img_gt - img_pred))))
                eval_dict['rec_psnr'].append(peak_signal_noise_ratio(img_gt, img_pred))
                eval_dict['rec_ssim'].append(structural_similarity(img_gt, img_pred, multichannel=True))

                # --- Depth
                depth_gt = np.expand_dims(input_batch['depth_gt'][b, n], axis=-1)  # (H, W, 1)
                depth_pred = output['depth_pred'][b, n]

                eval_dict['d_abs-rel-diff'].append(np.mean(np.abs(depth_pred - depth_gt) / depth_gt))
                eval_dict['d_squ-rel-diff'].append(np.mean(np.square(depth_pred - depth_gt) / depth_gt))
                eval_dict['d_rmse'].append(np.sqrt(np.mean(np.square(depth_pred - depth_gt))))

                # --- Instance
                gt_msk = np.expand_dims(input_batch['msk_gt'][b, :, n], axis=-1)  # (N_obj, H, W, 1)
                pred_msk = output['obj_msk_pred'][b, :, n]
                min_inst_pixel = 25
                prec, rec, _, _, _ = inst_ap(gt_msk, pred_msk, th=0.5, min_n_pxls=min_inst_pixel)
                eval_dict['inst_ap0.5'].append(prec)
                eval_dict['inst_ar0.5'].append(rec)
                tmp_ap_ist = []
                for th in np.arange(0.5, 1., 0.05):
                    inst_res = inst_ap(gt_msk, pred_msk, th, min_n_pxls=min_inst_pixel)
                    tmp_ap_ist.append(inst_res[0])
                eval_dict['inst_ap'].append(np.mean(tmp_ap_ist))

        for k, v in losses.items():
            if k[:1] == 'l':
                if k not in loss_dict:
                    loss_dict[k] = 0.
                loss_dict[k] += v/num_batches

    if summary_writer is not None:
        # last batch only for visualization
        losses_batch = {}
        for k, v in losses.items():
            if k[0] != 'l':
                losses_batch[k] = v
        batch_data = {**input_batch, **output, **losses_batch}

        if is_training:
            tb_funcs.summarize_all(summary_writer, epoch, loss_dict=loss_dict, data_dict=batch_data, params=params,
                                   imgs_types=['rgb_in', 'rgb_pred', 'rgb_pred_refine', 'normal_pred', 'rgb_pred_gauss',
                                               'tmp_depth_diff',
                                               'obj_rgb_pred', 'obj_depth_pred', 'obj_msk_pred'], log_file=LOG_FILE)
        else:
            loss_dict_tmp = {
                'loss_total': loss_dict['loss_total']
            }
            tb_funcs.summarize_all(summary_writer, epoch, loss_dict=loss_dict_tmp, eval_dict=eval_dict,
                                   log_file=LOG_FILE, mode='val')

    if is_training and np.isnan(loss_dict['loss_total']):
        LOG_FILE.write('NaN error! Break Training.')
        exit(1)


# --------------------------------------------------
# ---


if __name__ == "__main__":

    FLAGS = parse_arguments()

    if FLAGS.message == "":
        print('Need to specify message/ name for  model [--message]')
        exit(0)
    model_base_name, model_ext_name = FLAGS.model.split('-')

    if FLAGS.deepsdf_dir == "":
        print('Need to specify pre-trained DeepSDF model  model [--deepsdf_dir]')
        exit(0)
    if not os.path.exists(os.path.join(FLAGS.deepsdf_dir, 'ckpts', 'checkpoint')):
        print('No DeepSDF model found at {}'.format(FLAGS.deepsdf_dir))
        exit(0)
    DEEPSDF_DIR = FLAGS.deepsdf_dir

    if FLAGS.config == "":
        print('Need to specify config file [--config]')
        exit(0)

    if FLAGS.no_gpu == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # Load model and config file
    LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.dataset + '_' + FLAGS.model + '(' + FLAGS.message + ')')
    # - Continue training
    if os.path.exists(os.path.join(LOG_DIR, 'ckpts', 'checkpoint')):
        model_file = os.path.join(LOG_DIR, model_base_name + '.py')
        model_module = load_module_from_log(model_base_name, model_file)

        cnfg_file = os.path.join(LOG_DIR, FLAGS.config + '.pkl')
        CONFIG = load_config(cnfg_file)

        new_training = False
    # - Start new training
    else:
        # Create all log dirs if they do not exist
        log_dirs = [FLAGS.log_dir, LOG_DIR]
        print(log_dirs)
        for log_dir in log_dirs:
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)

        model_file = os.path.join('models', model_base_name + '.py')
        model_module = importlib.import_module('.'+model_base_name, 'models')

        cnfg_file = os.path.join('config', FLAGS.config + '.py')
        config_module = importlib.import_module('.'+FLAGS.config, 'config')
        CONFIG = config_module.cnfg_dict

        # Back up files for later inspection
        proj_dir = os.path.dirname(os.path.abspath(__file__))
        backup_files = [model_file, "models/renderer.py", "train.py"]
        for backup_file in backup_files:
            backup_path = os.path.join(LOG_DIR, ntpath.split(backup_file)[-1])
            if os.path.exists(backup_path):
                os.remove(backup_path)
            copyfile(os.path.join(proj_dir, backup_file), backup_path)
        save_config(CONFIG, os.path.join(LOG_DIR, (ntpath.split(cnfg_file)[-1]).replace('.py', '.pkl')))

        new_training = True

    #CONFIG['training']['max_epoch'] = 800

    if 'anti_aliasing' not in CONFIG['model']['mosnet']:
        CONFIG['model']['mosnet']['anti_aliasing'] = False
    MODEL = model_module.get_model(CONFIG, model_ext_name)

    # load GT background data
    bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])

    if CONFIG['model']['mosnet']['anti_aliasing']:
        # The background depth map and image need to have the correct size:
        bg_img = tf.image.resize(bg_img, [128, 128])
        bg_depth = tf.image.resize(bg_depth, [128, 128])
        # With anti aliasing, we dont want any smoothing after epoch 200:
        CONFIG['model']['mosnet']['gauss_sigma'] = ['lin', 16./3, 0., 200]

    MODEL.set_gt_bg(depth=bg_depth)

    # Open Log-file
    LOG_FILE = LogFile(os.path.join(LOG_DIR, 'log_train.txt'))
    LOG_FILE.write(str(FLAGS))
    LOG_FILE.write('{:%d.%m_%H:%M}'.format(datetime.now()))
    AVG_LOSS_FILE = LogFile(os.path.join(LOG_DIR, 'log_train_avg_loss.txt'), new_training)  # TODO: write to file
    avg_loss_list = AVG_LOSS_FILE.read_lines()
    AVG_LOSS_LABELS = ['inst_ap0.5', 'inst_ar0.5', 'inst_ap',
                       'rec_rmse', 'rec_ssim', 'rec_psnr',
                       'd_abs-rel-diff', 'd_squ-rel-diff', 'd_rmse']

    # Load data
    print('Start loading data -', get_mem())
    DATA_TRAIN = data_provider.DataSetMultiObj(FLAGS.data_dir, 'train', CONFIG['data'])
    DATA_VAL = data_provider.DataSetMultiObj(FLAGS.data_dir, 'val', CONFIG['data'])
    print('Finished loading data -', get_mem())

    main()

    exit(0)
