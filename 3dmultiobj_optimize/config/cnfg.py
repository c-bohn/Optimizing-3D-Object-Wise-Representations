from termcolor import colored
from utils.shared_funcs import *


IMG_SIZE = 64
N_IMGS = 1

# -------------------------------------------------------------------------------------------
# Training
# -------------------------------------------------------------------------------------------

training = {'max_epoch': 400,
            'save_epoch': 25,
            'batch_size': 8,
            'learning_rate': 0.0001
            }

# -------------------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------------------

model = {'deepsdf': {'name': 'deepsdf',
                     'n_off_surface_samples': 1,
                     'n_on_surface_samples': 0,
                     'latent_prior_mean': 0.,
                     'latent_prior_stdvar': 0.1,
                     'dim_out': 1,
                     'dec_fc_layer': [64 for _ in range(4)],
                     'dec_latent_in': [2],
                     'loss-params': {'z-reg': 0.0005,
                                     'clamp_dist': 0.1,
                                     },
                     'n_latents': -1,
                     'dim_latent': -1,
                     },
         'deepcolor': {'name': 'deepsdf',
                       'n_off_surface_samples': 1,
                       'n_on_surface_samples': 0,
                       'n_latents': 0,
                       'dim_latent': 7,             # 8,
                       'dim_out': 3,
                       'dec_fc_layer': [64 for _ in range(4)],
                       'dec_latent_in': []
                       },
         'mosnet': {'name': 'mosnet',
                    'img_size': IMG_SIZE,
                    'n_imgs': N_IMGS,
                    'dim_latent': 21,               # 22,
                    'dim_latent_split': [8, 8, 5],  # [8, 8+1, 5],
                    'enc_convs': [32, 32, 64, 64],  # for higher resolution: [32, 32, 64, 64, 64, 64],
                    'enc_fc': [256, 64],
                    'gauss_kernel': 16,
                    'gauss_sigma': ['lin', 16./3, 0.5, 200],#['lin', 16./3, 0.5, int(training['max_epoch']*1./2.)],
                    'contrast_factor': 1.,
                    'l-weights': {
                        'rgb': 1.0,
                        'depth': 0.1,
                        'rgb_sil': 4000000., #['lin', 4000000., 0., 400, 500],
                        'depth_sil': 50., #['lin', 50., 0., 400, 500],
                        'z-reg': ['lin', 0.025, 0.0025, 400],#['lin', 0.025, 0.0025, int(training['max_epoch']*1.)],
                        'extr': 0.,  # 0.005,
                        'intersect': ['lin', 0., 0.001, 150, 200],  # 0.001,
                        'ground': 0.01,
                        'normal': ['lin', 0., 5., 150, 200],
                        # set whether to use a mask for the normal loss directly in the get_loss() function
                        'normal_sil': ['lin', 0., 10000000., 150, 200],
                        },
                    'anti_aliasing': False,
                    },
         }

# If we render with anti-aliasing, we don't want any blur applied to the render after 200 epochs. Since both the GT
# image and the predicted one are rendered with very similar anti-aliasing settings, they can be compared directly.
if model['mosnet']['anti_aliasing']:
    model['mosnet']['gauss_sigma'] = ['lin', 16./3, 0., 200]


# -------------------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------------------

data = {'img_size': IMG_SIZE,
        'n_samples': 4096,
        'max_n_obj': 1,
        'n_images': N_IMGS,
        'depth_max': 12.
        }


# -------------------------------------------------------------------------------------------
# Renderer
# -------------------------------------------------------------------------------------------

# - Camera Matrix, Blender
# std
euler = [1.0765, 0., 0.8549]
t = [3.74057, -3.25382, 2.67183]
f_64 = 70.

# # stacks (new)
# print(colored('[WARNING] cnfg.py, new camera callibration (stack scenes)', 'yellow'))
# euler = [1.3090, 0., 1.5708]
# t = [5.5, 0., 3.]
# f_64 = 60.


R = get_rotation_matrix(euler)
R, t = get_blender_cam(R, t)
K = [[(IMG_SIZE/64) * f_64, 0., IMG_SIZE/2.-0.5],  # (https://www.rojtberg.net/1601/from-blender-to-opencv-camera-and-back/)
     [0., (IMG_SIZE/64) * f_64, IMG_SIZE/2.-0.5],
     [0., 0., 1.]]

renderer = {'img_size': IMG_SIZE,
            'R': R,
            't': t,
            'K': K,
            'd_steps': 15,
            'd_max': 1.25,
            'depth_max': 12.}

if IMG_SIZE >= 128:
    renderer['d_steps'] = 25


# -------------------------------------------------------------------------------------------
# Dictionary Output
# -------------------------------------------------------------------------------------------

cnfg_dict = {
    'training': training,
    'model': model,
    'renderer': renderer,
    'data': data
}