source /nfs/hpc/share/nguyenkh/miniconda3/bin/activate
conda activate siamese


# run for all parts, #shots and runs at once

for part in {0..3}
do
    for k_shot in 1 5
    do
        for number in {0..4} # replace 4 here for test from 0 to 5 times
        do 
            echo ----------- part = $part k_shot = $k_shot number = $number ------------
            folder=FAPISv2_fcos_use_rf_mask_constrain_parts_unet_dist_part_$part
            echo $folder
            OMP_NUM_THREADS=7 CUDA_VISIBLE_DEVICES=0 python tools/test.py configs/FAPISv2_fcos_r50_caffe_fpn_gn_1x_4gpu.py \
            $folder/latest.pth --out $folder/results.pkl --eval bbox segm --part $part --k_shot $k_shot --number $number \
            --use_rf_mask --use_prototype --num_protos 16 --use_boundary --generate_weight --not_show_progress
        done
    done
done

