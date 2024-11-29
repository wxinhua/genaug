# CUDA_VISIBLE_DEVICES=1 python genaug.py \
#  --input /nfsroot/DATA/IL_Research/will/datasets/real_franka/read_h5/241015_pick_bread_plate_1/success_episodes/1015_101547/camera_right/data \
#  --output /nfsroot/DATA/IL_Research/will/datasets/aug_franka/241015_pick_bread_plate_1/success_episodes/1015_101547/camera_right \
#  --item "robot arm, bread, plate"

#!/bin/bash

# 基础输入路径和输出路径（请根据实际情况设置）
# base_input_path="/nfsroot/DATA/IL_Research/datasets/real_franka_1/h5_data/241015_pick_bread_plate_1/success_episodes"
# base_output_path="/nfsroot/DATA/IL_Research/will/datasets/test_h5"

# # 找到所有符合条件的目录，并按数字顺序排序
# all_trajectory_dirs=($(find "$base_input_path" -type d | grep -E '/1015_' | sort -t '_' -k 2 -n))

# # 确保至少有100条轨迹
# if [ ${#all_trajectory_dirs[@]} -lt 100 ]; then
#     echo "Error: Less than 100 valid trajectory directories found. Aborting."
#     exit 1
# fi

# # 从 all_trajectory_dirs 中提取前25个元素作为固定的列表
# trajectory_dirs=("${all_trajectory_dirs[@]:0:25}")

# # 循环10次
# for run in {1..10}; do
#     for input_dir in "${trajectory_dirs[@]}"; do
#         # 提取出时间戳部分
#         timestamp=$(basename "$input_dir")
        
#         # 定义要处理的相机列表
#         cameras=("camera_left" "camera_right" "camera_top")

#         # 处理每个相机方向
#         for camera in "${cameras[@]}"; do
#             # 构建可能的相机方向子目录路径
#             #camera_dir="$input_dir/$camera"
            
#             # 检查该相机方向的子目录是否存在
#             #if [ -d "$camera_dir" ]; then
#                 # 构建完整的输入和输出路径
#             full_input_path="$input_dir/data/trajectory.hdf5"
#             full_output_path="$base_output_path/$timestamp/"
#             #$camera/run_$run
#             # 确保输出目录存在
#             mkdir -p "$full_output_path"
            
#             # 执行Python脚本
#             CUDA_VISIBLE_DEVICES=0 python genaug.py \
#                 --input "$full_input_path" \
#                 --output "$full_output_path" \
#                 --item "robot arm, bread, plate" \
#                 --camera "$camera"
            
#         done
#     done
# done
export TRANSFORMERS_CACHE=/nfsroot/DATA/IL_Research/wk/huggingface_model
# 基础输入路径和输出路径（请根据实际情况设置）
base_input_path="/nfsroot/DATA/IL_Research/will/datasets/real_franka/read_h5/241015_pick_bread_plate_1/success_episodes"
#base_output_path="/nfsroot/DATA/IL_Research/will/datasets/aug_franka/241015_pick_bread_plate_1/success_episodes"
#### for test ###
base_output_path="/nfsroot/DATA/IL_Research/will/genaug-main/test_output"

# 找到所有符合条件的目录，并按数字顺序排序
all_trajectory_dirs=($(find "$base_input_path" -type d | grep -E '/1015_' | sort -t '_' -k 2 -n))

# 确保至少有100条轨迹
if [ ${#all_trajectory_dirs[@]} -lt 100 ]; then
    echo "Error: Less than 100 valid trajectory directories found. Aborting."
    exit 1
fi

# 从 all_trajectory_dirs 中提取前25个元素作为固定的列表
trajectory_dirs=("${all_trajectory_dirs[@]:0:175}")

# 循环10次
for run in {1..3}; do
    for input_dir in "${trajectory_dirs[@]}"; do
        # 提取出时间戳部分
        timestamp=$(basename "$input_dir")
        
        # 定义要处理的相机列表
        cameras=("camera_left" "camera_right" "camera_top")

        # 处理每个相机方向
        for camera in "${cameras[@]}"; do
            # 构建可能的相机方向子目录路径
            camera_dir="$input_dir/$camera"
            
            # 检查该相机方向的子目录是否存在
            if [ -d "$camera_dir" ]; then
                # 构建完整的输入和输出路径
                full_input_path="$camera_dir/data"
                full_output_path="$base_output_path/$timestamp/$camera/run_$run"
                
                # 确保输出目录存在
                mkdir -p "$full_output_path"
                
                # 执行Python脚本
                CUDA_VISIBLE_DEVICES=0 python genaug.py \
                    --input "$full_input_path" \
                    --output "$full_output_path" \
                    --item "robot arm, bread, plate" \
                    --camera "$camera"
            fi
        done
    done
done