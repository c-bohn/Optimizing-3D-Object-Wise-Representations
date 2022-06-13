import os
import argparse, json
import numpy as np


N_EXPERIMENTS = 5


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
    parser.add_argument('--exp', help='Name of experiment')
    parser.add_argument('--l', default=0, type=int, help='Line in avg. loss file (a.k.a. epoch bash)')
    FLAGS = parser.parse_args()
    return FLAGS


if __name__ == '__main__':

    FLAGS = parse_arguments()

    avg_loss_list = []

    for i in range(N_EXPERIMENTS):

        exp_dir = os.path.join(FLAGS.log_dir, FLAGS.exp.replace('xxx', str(i)))
        print(exp_dir)
        if not os.path.exists(exp_dir):
            print('Directory does not exist: ', exp_dir)
            continue

        avg_loss_path = os.path.join(exp_dir, 'log_train_avg_loss.txt')
        if not os.path.exists(avg_loss_path):
            print('No AVG loss information found in ', exp_dir)
            continue

        with open(avg_loss_path) as f:
            content = f.readlines()
            if len(content) == 0:
                continue
            if FLAGS.l == 0:
                n_ep = len(content)*25
            else:
                n_ep = FLAGS.l * 25
            content = content[FLAGS.l-1].split(' ')[1:]

            avg_loss_list.append([i, n_ep]+[float(x) for x in content])

    print('--------------------------------------')
    for l in avg_loss_list:
        print('[exp {}, {} eps] \n'
              '\t INST: ap:{:.2f}, ap_0.5:{:.2f}, ar_0.5:{:.2f};  \n'
              '\t REC: rmse:{:.2f}, psnr:{:.2f}, ssim:{:.2f};  \n'
              '\t DEPTH: rmse:{:.2f}, absrd:{:.2f}, sqrd:{:.2f}'.format(
            l[0], l[1], l[4], l[2], l[3], l[5], l[7], l[6], l[10], l[8], l[9]
        ))

    print('--------------------------------------')
    avg_l = np.mean(np.asarray(avg_loss_list), axis=0)
    print('[AVG, {:d} eps] \n'
          '\t INST: ap:{:.2f}, ap_0.5:{:.2f}, ar_0.5:{:.2f};  \n'
          '\t REC: rmse:{:.2f}, psnr:{:.2f}, ssim:{:.2f};  \n'
          '\t DEPTH: rmse:{:.2f}, absrd:{:.2f}, sqrd:{:.2f}'.format(
        int(avg_l[1]), avg_l[4], avg_l[2], avg_l[3], avg_l[5], avg_l[7], avg_l[6], avg_l[10], avg_l[8], avg_l[9]
    ))
