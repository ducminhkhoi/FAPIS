source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate siamese

export TORCH_HOME=/nfs/hpc/share/nguyenkh/torch


# for multiple GPUs training 

OMP_NUM_THREADS=8 ./tools/dist_train.sh configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu_dist.py 8 --work_dir work_dir/FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_0 --part 0 --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight

OMP_NUM_THREADS=8 ./tools/dist_train.sh configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu_dist.py 8 --work_dir work_dir/FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_2 --part 2 --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight

OMP_NUM_THREADS=8 ./tools/dist_train.sh configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu_dist.py 8 --work_dir work_dir/FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_3 --part 3 --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight

OMP_NUM_THREADS=8 ./tools/dist_train.sh configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu_dist.py 8 --work_dir work_dir/FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_4 --part 0 --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight


# for single GPU training 

OMP_NUM_THREADS=8 .python tools/train.py configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu.py 8 --work_dir work_dir/FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_0 --part 0 --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight

OMP_NUM_THREADS=8 .python tools/train.py configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu.py 8 --work_dir work_dir/FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_1 --part 1 --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight

OMP_NUM_THREADS=8 .python tools/train.py configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu.py 8 --work_dir work_dir/FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_2 --part 2 --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight

OMP_NUM_THREADS=8 .python tools/train.py configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu.py 8 --work_dir work_dir/FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_3 --part 3 --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight