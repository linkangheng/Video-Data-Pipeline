#bash

subsets=(
    aesthetics_plus_all_group_grayscale_all
    aesthetics_plus_all_group_bbox_all
    aesthetics_plus_all_group_normal_all
    aesthetics_plus_all_group_blur_all
    aesthetics_plus_all_group_hed_all
    aesthetics_plus_all_group_openpose_all
    aesthetics_plus_all_group_canny_all
    aesthetics_plus_all_group_hedsketch_all
    aesthetics_plus_all_group_outpainting_all
    aesthetics_plus_all_group_depth_all
    aesthetics_plus_all_group_inpainting_all
    aesthetics_plus_all_group_seg_all
)

for subset in ${subsets[@]}; do
    python /data/video_pack/pack/pack.py \
        --dataset unicontrol \
        --workers 64 \
        --type unicontrol \
        --save_path /mnt/jfs-test/data/unicontrol/tars/$subset \
        --total_machine 8 \
        --subset $subset \
        --machine_id $MACHINE_ID
done