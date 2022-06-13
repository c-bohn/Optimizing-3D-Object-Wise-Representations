from config.cnfg import *

MAX_N_OBJ = 3

cnfg_dict['model']['deepsdf']['n_latents'] = 25
cnfg_dict['model']['deepsdf']['dim_latent'] = 16

cnfg_dict['model']['mosnet']['n_slots'] = MAX_N_OBJ
cnfg_dict['model']['mosnet']['dim_latent'] = 29
cnfg_dict['model']['mosnet']['dim_latent_split'] = [16, 8, 5]
cnfg_dict['model']['mosnet']['size_range'] = [0.75, 1.25]

cnfg_dict['data']['max_n_obj'] = MAX_N_OBJ


assert(cnfg_dict['model']['mosnet']['dim_latent'] == sum(cnfg_dict['model']['mosnet']['dim_latent_split']))
assert(cnfg_dict['model']['deepcolor']['dim_latent']+1 == cnfg_dict['model']['mosnet']['dim_latent_split'][1])
