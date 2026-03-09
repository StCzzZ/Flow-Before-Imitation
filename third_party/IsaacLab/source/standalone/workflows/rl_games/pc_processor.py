import open3d as o3d
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



def farthest_point_sampling(self, point_cloud, num_points):
        
        sampled_points = o3d.geometry.PointCloud.farthest_point_down_sample(point_cloud, num_points)


        return sampled_points
