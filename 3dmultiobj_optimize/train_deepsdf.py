import os
import argparse
import ntpath
from datetime import datetime
from shutil import copyfile

# Disable TF info messages (needs to be done before tf import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from utils import data_provider
from utils.shared_funcs import *
from utils.tf_funcs import *
from utils import tb_funcs

# GPU should not be allocated entirely at beginning
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clevr', help='Data Set')
    parser.add_argument('--data_dir', help='Data dir')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--model', default='deepsdf-org', help='Model name [deepsdf-org, deepsdf-var]')
    parser.add_argument('--config', default='', help='Configuration file [cnfg_<model>_<dataset>]')
    parser.add_argument('--message', default='', help='Message that specifies settings etc. (for log dir name)')
    parser.add_argument('--no_gpu', default='0', help='Do not use GPU')
    return parser.parse_args()


# --------------------------------------------------
# --- Main Training

def main():
    time_start = datetime.now()
    time_start_str = '{:%d.%m_%H:%M}'.format(datetime.now())

    # Create dataset iterator for training and validation
    dataset_train, iterator_train = get_data_iterator(DATA_TRAIN, CONFIG['training'], type='shapedec', shuffle=True)
    iterators = {
        'train': iterator_train
    }

    print("+++++++++++++++++++++++++++++++")

    # Training parameters
    params = {'w_z-reg': CONFIG['model']['deepsdf']['loss-params']['z-reg'],
              'clamp_dist': CONFIG['model']['deepsdf']['loss-params']['clamp_dist']}

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=CONFIG['training']['learning_rate'])

    # Operator to save and restore all the variables.
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=MODEL)
    manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)

    # Writer for summary
    writer = tf.summary.create_file_writer(os.path.join(LOG_DIR, 'summary'))

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        ckpt.step.assign_add(1)
        start_epoch = int(ckpt.step)
        print("Restored from {}, epoch {}".format(manager.latest_checkpoint, start_epoch-1))
    else:
        print("Initializing from scratch.")
        start_epoch = 1

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
        time_delta = datetime.now()-time_start
        LOG_FILE.write('**** EPOCH %03d - start: %s - now: %s ****' % (epoch, time_start_str, time_now_str))

        run_one_epoch(epoch, MODEL, ops, True, writer)

        if int(ckpt.step) % CONFIG['training']['save_epoch'] == 0 or epoch == max_epoch:
            save_path = manager.save()
            LOG_FILE.write("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
        ckpt.step.assign_add(1)

        # For cluster: break training after some time
        time_delta_hours, time_delta_minutes, time_delta_seconds = split_time_delta(time_delta)
        print('Runtime: {} day(s), {} hour(s), {} minute(s), {} seconds'.format(time_delta.days, time_delta_hours,
                                                                                time_delta_minutes, time_delta_seconds))


def train_step(net, data, optimizer, params):
    """Trains 'net' on 'example' using 'optimizer' wrt 'params'."""
    with tf.GradientTape() as tape:
        output = net([data['pnts_coord'], data['scene_ids']])
        losses = net.get_loss(output, data, params)
    variables = net.trainable_variables
    gradients = tape.gradient(losses['loss_total'], variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return output, losses


def run_one_epoch(epoch, net, ops, is_training, summary_writer=None):
    LOG_FILE.write('----')

    num_batches = (DATA_TRAIN.get_size() // CONFIG['training']['batch_size']) + 1

    loss_dict = {}

    for batch_id in range(num_batches):

        input_batch = next(ops['iterators']['train'])
        input_batch = net.get_input(input_batch)
        output, losses = train_step(net, input_batch, ops['optimizer'], ops['params'])

        # net.summary()

        for k, v in losses.items():
            if 'l_' in k or 'loss' in k:
                if k not in loss_dict:
                    loss_dict[k] = 0.
                loss_dict[k] += float(v) / num_batches

    # Image visualizations (last batch only)
    batch_data = {**input_batch, **output}
    img_size = batch_data['rgb_gt'].shape[2]
    slice_list = []
    for a in ['x', 'y', 'z']:
        slice_coords = gen_slice(a, size=img_size)
        output_slices = net([slice_coords, batch_data['scene_ids'][0:1]])
        pred_slice = tf.reshape(output_slices['sdf'], [1, img_size, img_size, 1])
        slice_list.append(pred_slice)
    batch_data['slice_pred'] = tf.stack(slice_list, axis=1)  # (BS'=1, 3, H, W, 1)

    tb_funcs.summarize_all(summary_writer, epoch, loss_dict=loss_dict, data_dict=batch_data,
                           imgs_types=['rgb_gt', 'slice_gt', 'slice_pred'], log_file=LOG_FILE)

    if is_training and np.isnan(loss_dict['loss_total']):
        LOG_FILE.write('NaN error! Break Training.')
        exit(1)


# --------------------------------------------------


if __name__ == "__main__":

    FLAGS = parse_arguments()

    if FLAGS.message == "":
        print('Need to specify message/ name for  model [--message]')
        exit()
    model_base_name, model_ext_name = FLAGS.model.split('-')

    if FLAGS.config == "":
        FLAGS.config = 'cnfg_' + model_base_name + '_' + FLAGS.dataset

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
        backup_files = [model_file, "train_deepsdf.py"]
        for backup_file in backup_files:
            backup_path = os.path.join(LOG_DIR, ntpath.split(backup_file)[-1])
            if os.path.exists(backup_path):
                os.remove(backup_path)
            copyfile(os.path.join(proj_dir, backup_file), backup_path)
        save_config(CONFIG, os.path.join(LOG_DIR, (ntpath.split(cnfg_file)[-1]).replace('.py', '.pkl')))

    MODEL = model_module.get_model(CONFIG['model']['deepsdf'], model_ext_name)

    # Open Log-file
    LOG_FILE = LogFile(os.path.join(LOG_DIR, 'log_train.txt'))
    LOG_FILE.write(str(FLAGS))
    LOG_FILE.write('{:%d.%m_%H:%M}'.format(datetime.now()))

    # Load data
    print('Start loading data -', get_mem())
    DATA_TRAIN = data_provider.DataSetShapeDec(FLAGS.data_dir, 'train', CONFIG['data'])
    print('Finished loading data -', get_mem())

    main()

    exit(0)
