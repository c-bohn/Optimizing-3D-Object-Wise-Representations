#!/bin/bash


# Set in the eval script if this is a plain or an r-and-c run
# If r-and-c, make sure that each model has the correct latent file in its eval directory as well as the current mosnet.py

#echo "starting in 4 hours..."
#sleep 14400   #= 4h
#echo "starting now"

model_msgs=(    train_mosnet_optimized_sil_400_ep_rep_3
                train_mosnet_optimized_sil_400_ep_rep_4
                train_mosnet_optimized_sil_400_ep_rep_3_eval_w_r_and_c
                train_mosnet_optimized_sil_400_ep_rep_4_eval_w_r_and_c
              )

for i in ${model_msgs[@]}
  do
    echo "\n\n----------------------------"
    echo "Eval run for model"
    echo $i
    echo "\n"
    python eval.py --config cnfg_mosnet-org_obj3_clevr --data_dir ../data_basic/clevr_christian --dataset clevr --message $i --model mosnet-org
  done