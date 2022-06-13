from config.cnfg import *

cnfg_dict['training']['max_epoch'] = 5000
cnfg_dict['training']['save_epoch'] = 1000
cnfg_dict['training']['batch_size'] = 64
cnfg_dict['training']['learning_rate'] = 0.001

cnfg_dict['model']['deepsdf']['n_off_surface_samples'] = 3277
cnfg_dict['model']['deepsdf']['n_on_surface_samples'] = 819
cnfg_dict['model']['deepsdf']['n_latents'] = 27
cnfg_dict['model']['deepsdf']['dim_latent'] = 8
