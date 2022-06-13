import os
import argparse
import ntpath
import copy
from datetime import datetime
from shutil import copyfile

from utils import data_provider
from utils.shared_funcs import *
from utils.tf_funcs import *

from models import mosnet

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='clevr', help='Data Set')
    parser.add_argument('--deepsdf_dir', help='path to pre-trained DeepSDF Model basedir')
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--data_dir', help='Data dir')
    parser.add_argument('--model', default='mosnet-org', help='Model name [mosnet-<variant>]')
    parser.add_argument('--config', default='', help='Configuration file [cnfg_<model>_obj<#>_<dataset>]')
    parser.add_argument('--message', default='', help='Message that specifies settings etc. (for log dir name)')
    return parser.parse_args()

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

    MODEL = model_module.get_model(CONFIG, model_ext_name)
    # load GT background data
    bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])
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
        print('ERROR: Could not restore the model')
        exit(1)

    MODEL_w_refiner = mosnet.MultiObj3DNet_w_refiner(CONFIG)

    bg_img, bg_depth = data_provider.load_bg_img(FLAGS.data_dir, CONFIG['data']['img_size'])
    MODEL_w_refiner.set_gt_bg(depth=bg_depth)

    MODEL_w_refiner.encoder_bg = copy.deepcopy(MODEL.encoder_bg)
    MODEL_w_refiner.encoder_obj = copy.deepcopy(MODEL.encoder_obj)
    MODEL_w_refiner.decoder_sdf = copy.deepcopy(MODEL.decoder_sdf)
    MODEL_w_refiner.decoder_rgb = copy.deepcopy(MODEL.decoder_rgb)
    MODEL_w_refiner.renderer = copy.deepcopy(MODEL.renderer)




    #######################
    # #Just testig:
    #
    # CONFIG['r-and-c'] = {}
    # for k, v in CONFIG['training'].items():
    #     CONFIG['r-and-c'][k] = v
    #
    # # Create dataset iterator over the validation set
    # dataset_val, iterator_val = get_data_iterator(DATA_VAL, CONFIG['r-and-c'])
    # iterator = iterator_val
    #
    # input = next(iterator)
    # input = MODEL.get_input(input)
    #
    # rgb_in = input['rgb_in']
    #
    # results = MODEL_w_refiner([input['rgb_in']], training=False)
    #
    #
    # rgb_pred = results['rgb_pred'][0]
    # rgb_pred = (np.transpose(rgb_pred, (1, 0, 2, 3))).reshape((64, 64, 3))
    # from utils import viz
    # viz.show_image(rgb_pred, './model_cnversion_test_result.png')
    # #######################




    # Redefine ckpt and manager to save the model with refiner:
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=MODEL_w_refiner)
    LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.dataset + '_' + model_base_name + '-refine' + '(' + FLAGS.message + ')')
    manager = tf.train.CheckpointManager(ckpt, os.path.join(LOG_DIR, 'ckpts'), max_to_keep=3)
    save_path = manager.save()
    ckpt.step.assign_add(1)

    print("Done, saved new model in {}".format(LOG_DIR))

    #This script does not copy the other files that need to be in a model folder, simply copy them manually from the source model

    exit(0)