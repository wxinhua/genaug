import h5py
import os
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import argparse
import glob
import re

class ReadH5Files():
    def __init__(self, robot_infor):
        self.camera_names = robot_infor['camera_names']
        self.camera_sensors = robot_infor['camera_sensors']

        self.arms = robot_infor['arms']
        self.robot_infor = robot_infor['controls']
        # 'joint_velocity_left', 'joint_velocity_right',
        # 'joint_effort_left', 'joint_effort_right',
        pass

    def decoder_image(self, camera_rgb_images, camera_depth_images):
        if type(camera_rgb_images[0]) is np.uint8:
            rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            return rgb, depth
        else:
            rgb_images = []
            depth_images = []
            for idx, camera_rgb_image in enumerate(camera_rgb_images):
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                if camera_depth_images is not None:
                    depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                else:
                    depth = np.asarray([])
                rgb_images.append(rgb)
                depth_images.append(depth)
            rgb_images = np.asarray(rgb_images)
            depth_images = np.asarray(depth_images)
            return rgb_images, depth_images
        
    def read_aug_images(self, aug_path):
        aug_images = []
        
        # 使用正则表达式提取编号
        def extract_number(file_name):
            match = re.search(r'_(\d+)\.png$', file_name)
            if match:
                return int(match.group(1))
            else:
                return -1  # 如果没有匹配到编号，返回-1
        
        # 获取文件名列表并按编号排序
        image_names = sorted(os.listdir(aug_path), key=extract_number)
        
        for image_name in image_names:
            if image_name.endswith('.png'):
                image_path = os.path.join(aug_path, image_name)
                aug_image = cv2.imread(image_path)
                if aug_image is None:
                    print(f"Failed to read image: {image_path}")
                    continue
                aug_image = cv2.imencode(".jpg", aug_image)[1]
                aug_images.append(aug_image)
        
        return aug_images
    
    def read_aug_images_matrix(self, image_list):
        aug_images_list= []
        for image in image_list:
            aug_image = cv2.imencode(".jpg", image)[1]
            aug_image = np.asarray(aug_image)
            aug_images_list.append(aug_image)
        return aug_images_list

    def create_new_h5_file(self, src_root, new_file_path ):
        # 创建一个新的 HDF5 文件并复制原始文件的结构
        with h5py.File(new_file_path, 'w') as dst_root:
            def copy_group(name, obj):
                if isinstance(obj, h5py.Group):
                    dst_root.create_group(name)
                elif isinstance(obj, h5py.Dataset):
                    dst_root.create_dataset(name, data=obj[:])
                    #dst_root.copy(obj, name)

            src_root.visititems(copy_group)

            # 复制属性
            for attr_key in src_root.attrs:
                dst_root.attrs[attr_key] = src_root.attrs[attr_key]
        

    def execute(self, file_path, camera_frame=None, control_frame=None):
        with h5py.File(file_path, 'r') as root:
            print(file_path)
            #is_sim = root.attrs['sim']
            is_compress = root.attrs['compress']
            print(f'is_compress: {is_compress}' )
            is_compress = True
            # print('is_compress:',T)
            # select camera frame id
            image_dict = defaultdict(dict)

            # data_save_dir = os.path.join(save_dir, 'data')
            # if not os.path.exists(data_save_dir):
            #     os.makedirs(data_save_dir)

            # rgb_save_dir = os.path.join(save_dir, 'rgb')
            # if not os.path.exists(rgb_save_dir):
            #     os.makedirs(rgb_save_dir)

            # depth_color_save_dir = os.path.join(save_dir, 'depth_color')
            # if not os.path.exists(depth_color_save_dir):
            #     os.makedirs(depth_color_save_dir)  

            # depth_save_dir = os.path.join(save_dir, 'depth')
            # if not os.path.exists(depth_save_dir):
            #     os.makedirs(depth_save_dir)

            for cam_name in self.camera_names:
                if is_compress:
                    if camera_frame is not None:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][camera_frame])
                        # decode_rgb, decode_depth = self.decoder_image(
                        #     camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                        #     camera_depth_images=None)

                        for i in range(len(decode_rgb)):
                            
                            # cv2.imwrite(os.path.join(data_save_dir, f'rgb_{cam_name}_{i}.jpg'),
                            #             decode_rgb[i][...,::-1])
                            rgb_image = cv2.cvtColor(decode_rgb[i], cv2.COLOR_BGR2RGB)
                            # Image.fromarray(rgb_image).save(os.path.join(data_save_dir, f'rgb_{cam_name}_{i}.jpg'))
                            # cv2.imwrite(os.path.join(data_save_dir, f'rgb_{cam_name}_{i}.jpg'),
                            #             decode_rgb[i])
                        for i in range(len(decode_depth)):
                            # print('decode_depth[i]:',decode_depth[i],decode_depth[i].shape,np.max(decode_depth[i]))
                            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(decode_rgb[i], alpha=0.07),
                                                               cv2.COLORMAP_JET)
                            
                            # cv2.imwrite(os.path.join(data_save_dir, f'depth_{depth_colormap}_{i}.png'),
                            #             depth_colormap)
                            # np.save(os.path.join(data_save_dir, f'{cam_name}_{i}.npy'), decode_depth[i])
                    else:
                        # print(f"cam_name: {cam_name}")
                        # print(f"camera_sensors: {self.camera_sensors[0]}, {self.camera_sensors[1]}")
                        # tmp = root['observations'][self.camera_sensors[0]][cam_name]
                        # tmp1 = root['observations']
                        # tmp2 = root['observations'][self.camera_sensors[0]]
                        # print(f"tmp1: {tmp1}")
                        # print(f"tmp2: {tmp2}")
                        # print(f"tmp2 key: {tmp2.keys()}")
                        # tmp3 = root['observations'][self.camera_sensors[0]][cam_name]
                        # print(f"tmp3: {tmp3}")
                        # print(f"Ori Data set shape: {root['observations'][self.camera_sensors[0]][cam_name].shape}")
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][:],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][:])
            return decode_rgb, decode_depth
                        # decode_rgb, decode_depth = self.decoder_image(
                        #     camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][:],
                        #     camera_depth_images=None)
                        # for i in range(len(decode_rgb)):
                            # cv2.imwrite(os.path.join(data_save_dir, f'rgb_{cam_name}_{i}.jpg'),
                            #             decode_rgb[i][...,::-1])
                            # cv2.imwrite(os.path.join(data_save_dir, f'rgb_{cam_name}_{i}.jpg'),
                            #             decode_rgb[i])
                            # rgb_image = cv2.cvtColor(decode_rgb[i], cv2.COLOR_BGR2RGB)
                            # Image.fromarray(rgb_image).save(os.path.join(data_save_dir, f'rgb_{cam_name}_{i}.jpg'))
                        # for i in range(len(decode_depth)):
                            # print('decode_depth[i]:',decode_depth[i],decode_depth[i].shape,np.max(decode_depth[i]))
                            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(decode_depth[i], alpha=0.07),
                            #                                    cv2.COLORMAP_JET)
                            # cv2.imwrite(os.path.join(data_save_dir, f'depth_{cam_name}_{i}.png'),
                            #             depth_colormap)
                            # np.save(os.path.join(data_save_dir, f'{cam_name}_{i}.npy'), decode_depth[i])

        #             image_dict[self.camera_sensors[0]][cam_name] = decode_rgb
        #             # image_dict[self.camera_sensors[1]][cam_name] = decode_depth

        #         else:
        #             if camera_frame:
        #                 image_dict[self.camera_sensors[0]][cam_name] = root[
        #                     'observations'][self.camera_sensors[0]][cam_name][camera_frame]
        #                 # image_dict[self.camera_sensors[1]][cam_name] = root[
        #                 #     'observations'][self.camera_sensors[1]][cam_name][camera_frame]
        #             else:
        #                 image_dict[self.camera_sensors[0]][cam_name] = root[
        #                    'observations'][self.camera_sensors[0]][cam_name][:]
        #                 # image_dict[self.camera_sensors[1]][cam_name] = root[
        #                 #    'observations'][self.camera_sensors[1]][cam_name][:]


        #     control_dict = defaultdict(dict)
        #     for arm_name in self.arms:
        #         for control in self.robot_infor:
        #             if control_frame:
        #                 control_dict[arm_name][control] = root[arm_name][control][control_frame]
        #             else:
        #                 control_dict[arm_name][control] = root[arm_name][control][:]
        #     # print('infor_dict:',infor_dict)
        #     base_dict = defaultdict(dict)
        # # print('control_dict[puppet]:',control_dict['master']['joint_position_left'][0:1])
        # return image_dict, control_dict, base_dict, is_sim, is_compress
    
    def check_original_file(self, file_path, cam_name):
        with h5py.File(file_path, 'r') as src_root:
            print(f"Original file dataset shape: {src_root['observations'][self.camera_sensors[0]][cam_name].shape}")

    def write_aug_images_to_h5(self, new_file_path, cam_name, aug_rgb_images, camera_frame=None):
        # 将增强后的图像数据写入新HDF5文件
        with h5py.File(new_file_path, 'r+') as dst_root:
            if camera_frame is not None:
                for i in range(len(aug_rgb_images)):

                    dst_root['observations'][self.camera_sensors[0]][cam_name][i] = aug_rgb_images[i]
                
            else:
                for i in range(len(aug_rgb_images)):
                    dst_root['observations'][self.camera_sensors[0]][cam_name][i] = aug_rgb_images[i]

    # def write_aug_images_to_h5(self, new_file_path, cam_name, aug_rgb_images, camera_frame=None):
    # # 将增强后的图像数据写入新HDF5文件
    #     with h5py.File(new_file_path, 'r+') as dst_root:
    #         dataset_path = f'observations/{self.camera_sensors[0]}/{cam_name}'
            
    #         # 检查数据集是否存在，如果存在且形状不正确，则删除
    #         if dataset_path in dst_root:
    #             existing_dataset = dst_root[dataset_path]
    #             if existing_dataset.shape != (len(aug_rgb_images), *aug_rgb_images[0].shape):
    #                 del dst_root[dataset_path]
    #                 print(f"Deleted existing dataset at {dataset_path} due to shape mismatch.")
            
    #         # 创建数据集
    #         if dataset_path not in dst_root:
    #             dst_root.create_dataset(
    #                 dataset_path,
    #                 shape=(len(aug_rgb_images), *aug_rgb_images[0].shape),
    #                 dtype=aug_rgb_images[0].dtype
    #             )
            
    #         if camera_frame is not None:
    #             for i in range(len(aug_rgb_images)):
    #                 print(f"Data set shape: {dst_root[dataset_path].shape}")
    #                 print(f"Image shape: {aug_rgb_images[i].shape}")
    #                 dst_root[dataset_path][i] = aug_rgb_images[i]
    #         else:
    #             for i in range(len(aug_rgb_images)):
    #                 dst_root[dataset_path][i] = aug_rgb_images[i]
                        
        
    def process_aug_images(self, file_path, save_dir, output_dir, camera_frame=None):
        new_file_path = os.path.join(output_dir, 'aug_data.h5')

        # 读取原始文件
        with h5py.File(file_path, 'r') as src_root:
            # 创建新文件并复制结构
            self.create_new_h5_file(src_root, new_file_path)

            # 读取增强图像并写入新HDF5文件
            for cam_name in self.camera_names:
                cam_dir = os.path.join(save_dir, cam_name)
                if os.path.exists(cam_dir) and os.path.isdir(cam_dir):
                    run_dirs = [d for d in os.listdir(cam_dir) if d.startswith('run_')]
                    for run_dir in run_dirs:
                        run_id = int(run_dir.split('_')[1])
                        if camera_frame is not None and run_id != camera_frame:
                            continue
                        aug_path = os.path.join(cam_dir, run_dir)
                        aug_rgb_images = self.read_aug_images(aug_path)
                        self.write_aug_images_to_h5(new_file_path, cam_name, aug_rgb_images, run_id)

    def process_aug_images_matrix(self, new_rgb, file_path, output_dir):
        new_file_path = os.path.join(output_dir, 'aug_data.h5')

        # 读取原始文件
        with h5py.File(file_path, 'r') as src_root:
            # 创建新文件并复制结构
            self.create_new_h5_file(src_root, new_file_path)

            # 读取增强图像并写入新HDF5文件
            for cam_name in self.camera_names:
                aug_rgb_images = self.read_aug_images_matrix(new_rgb)
                self.write_aug_images_to_h5(new_file_path, cam_name, aug_rgb_images)
                        


def str_to_list(camera_names_str):
    # 使用逗号和空格来分割字符串，并去除每个元素的前后空白
    camera_names = [name.strip() for name in camera_names_str.split(',')]
    return camera_names

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Read and process HDF5 files.")
    parser.add_argument('--camera_names', type=str, required=True, help='Comma-separated list of camera names')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--new_path', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the processed data')

    args = parser.parse_args()
    camera_names = args.camera_names
    file_path = args.file_path
    save_dir = args.save_dir
    new_path = args.new_path

    camera_names = str_to_list(camera_names)

    
    os.makedirs(save_dir, exist_ok=True)
    
    # robot_infor = {'camera_names': ['camera_left', 'camera_right', 'camera_top'],
    #                'camera_sensors': ['rgb_images','depth_images'],
    #                'arms': ['master', 'puppet'],
    #                'controls': ['joint_position']}

    robot_infor = {'camera_names': camera_names,
                   'camera_sensors': ['rgb_images','depth_images'],
                   'arms': ['master', 'puppet'],
                   'controls': ['joint_position']}

    

    read_h5files = ReadH5Files(robot_infor)
    image_dict, control_dict, base_dict, is_sim, is_compress = read_h5files.execute(file_path=file_path, save_dir=save_dir)
    
    read_h5files.process_aug_images(save_dir=save_dir, file_path=file_path, output_dir=new_path)