#!/bin/bash

WORK_DIR="/workspace/dino"
cd $WORK_DIR


# Loop over several ViT-Base models with different MLP heads

# declare -a N_LAYERS=("1" "2")
# declare -a N_NODES=("40" "400")
# declare -a HIDDEN_ACT=("gelu" "relu")
declare -a TYPE=("linear" "mlp")
declare -a N_LAYERS=("1")
declare -a N_NODES=("40")
declare -a HIDDEN_ACT=("relu")
declare -a LOSS=("mape" "mse")

for type in "${TYPE[@]}"
do
    for n_layers in "${N_LAYERS[@]}"
    do
        for n_nodes in "${N_NODES[@]}"
        do
            for hidden_act in "${HIDDEN_ACT[@]}"
            do
                for loss in "${LOSS[@]}"
                do
                    # echo -n "dummy.py "
                    # echo -n "--n_hidden_layers $n_layers "
                    # echo "--n_hidden_nodes $n_nodes "

                    # python dummy.py \
                    # --n_hidden_layers $n_layers \
                    # --n_hidden_nodes $n_nodes

                    if test "$type" = 'linear'
                    then
                        OUTPUT_NAME=weight_head_base_"$type"_LOSS"$loss"
                    else
                        # Assuming MLP head
                        OUTPUT_NAME=weight_head_base_"$type"_L"$n_layers"_N"$n_nodes"_HIDACT"$hidden_act"_LOSS"$loss"
                    fi

                    mkdir ./$OUTPUT_NAME

                    # python dummy.py \
                    python -m torch.distributed.launch --nproc_per_node=4 eval_weight.py \
                    --arch vit_base \
                    --n_last_blocks 1 \
                    --patch_size 8 \
                    --avgpool_patchtokens True \
                    --batch_size_per_gpu 73 \
                    --epoch 500 \
                    --lr 0.0001 \
                    --data_path_cows /workspace/Downloads/scaled_segmented_train_val_sets/ \
                    --data_path_fields /workspace/Downloads/Flickr_fields_train_val_sets/ \
                    --csv_weight_filename /workspace/Downloads/video_info_2021-Feb-Mar.csv \
                    --num_workers 8 \
                    --output_act linear \
                    --head_type $type \
                    --hidden_act $hidden_act \
                    --n_hidden_layers $n_layers \
                    --n_hidden_nodes $n_nodes \
                    --loss $loss \
                    --output_dir ./$OUTPUT_NAME \
                    2>/dev/null

                    echo ""

                    # Tar and copy results to AWS S3
                    tar -cvf $OUTPUT_NAME.tar ./$OUTPUT_NAME
                    aws s3 cp $OUTPUT_NAME.tar s3://bicog-datasets/cows/$OUTPUT_NAME.tar
                done
            done
        done
    done
done
