import open3d as o3d
import torch
import numpy as np
from omni.isaac.lab.sensors.camera.utils import create_pointcloud_from_rgbd


class PointcloudVisualizer() :
	def __init__(self) -> None:
		self.vis = o3d.visualization.VisualizerWithKeyCallback()
		self.vis.create_window()
		# self.vis.register_key_callback(key, your_update_function)
	
	def add_geometry(self, cloud) :
		self.vis.add_geometry(cloud)

	def update(self, cloud):
		#Your update routine
		self.vis.update_geometry(cloud)
		self.vis.update_renderer()
		self.vis.poll_events()



class PointcloudGenerator():

    def __init__(self) -> None:
        pass
	
    def farthest_point_sampling(self, point_cloud, num_points):
            """
            使用Open3D库进行最远点采样

            参数：
                point_cloud (numpy.array): 表示点云的NumPy数组，每一行是一个点的坐标 [x, y, z]
                num_points (int): 采样点的数量

            返回：
                numpy.array: 采样后的点云数组，每一行是一个采样点的坐标 [x, y, z]
            """
            sampled_points = o3d.geometry.PointCloud.farthest_point_down_sample(point_cloud, num_points)


            return sampled_points

    def get_pc_and_color(self, obs, obs_id, camera_numbers, num_points):
        points_all = []
        colors_all = []
        for cam_id in range(camera_numbers):
            rgba_all = obs.get(f"rgba_img_0{cam_id}", None)
            depth_all = obs.get(f"depth_img_0{cam_id}", None)
            intrinsic_matrices_all = obs.get(f"intrinsic_matrices_0{cam_id}", None)
            pos_w_all = obs.get(f"pos_w_0{cam_id}", None)
            quat_w_ros_all = obs.get(f"quat_w_ros_0{cam_id}", None)

            rgba = rgba_all[obs_id]
            depth = depth_all[obs_id]
            intrinsic_matrix = intrinsic_matrices_all[obs_id]
            pos_w = pos_w_all[obs_id]
            quat_w_ros = quat_w_ros_all[obs_id]

            # generate point cloud
            points_xyz, points_rgb = create_pointcloud_from_rgbd(
                intrinsic_matrix=intrinsic_matrix,
                depth=depth,
                rgb=rgba,
                normalize_rgb=True,  # normalize to get 0~1 pc, the same as dp3
                position=pos_w,
                orientation=quat_w_ros,
            )

            # add points and colors to list
            points_all.append(points_xyz)
            colors_all.append(points_rgb)

        # concatenate points and colors
        points_all = torch.cat(points_all, dim=0)
        colors_all = torch.cat(colors_all, dim=0)

        # sample points and colors. 

        points_env = o3d.geometry.PointCloud()
        points_env.points = o3d.utility.Vector3dVector(points_all.detach().cpu().numpy())
        points_env.colors = o3d.utility.Vector3dVector(colors_all.detach().cpu().numpy())

        points_env = self.farthest_point_sampling(points_env, num_points)
        
        points_cropped, colors_cropped = np.asarray(points_env.points), np.asarray(points_env.colors)

        return points_cropped, colors_cropped