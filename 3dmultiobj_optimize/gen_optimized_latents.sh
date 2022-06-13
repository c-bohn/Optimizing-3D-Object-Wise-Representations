#!/bin/bash


# Set in the render-and-compare script if anti-aliasing should be applied in the optimization
# For each model <model> in model_msgs also a model <model>_eval_w_r_and_c / <model>_eval_w_r_and_c_anti_aliasing_depth_w_aa needs to already exist
# Make sure that no latent file already exists in these directories

model_msgs=(  train_mosnet_optimized_sil_400_ep_rep_3
              train_mosnet_optimized_sil_400_ep_rep_4
              )

for model_msg in ${model_msgs[@]}
  do
    for i in $(seq 0 10 990)
      do
        python render_and_compare.py --dataset clevr --data_dir ../data_basic/clevr_christian --model mosnet-org --config cnfg_mosnet-org_obj3_clevr --message $model_msg --split val --start_scene $i --stop_scene $((i+10))
      done
  done


