import open3d as o3d
import numpy as np

# points_data = np.loadtxt("E:/vsproject/勋仪交接/标克艾芬达暖水管/测量主管尺寸/vp/11pnts.txt", delimiter=",", dtype=np.float32)
points_data = np.loadtxt("E:/vsproject/勋仪交接/标克艾芬达暖水管/测量主管尺寸/vp/11pnts.txt", delimiter=",")
print('shape: ', points_data.shape)
print('data type: ', points_data.dtype)

points_data[:,2] = points_data[:,2] * 21 # 将所有行的第三列乘以 21

# points_data = np.random.rand(100, 3)  # 示例数据：100 个随机 3D 点
colors_data = np.random.rand(100, 3)  # 100 个随机 RGB 颜色数组

pcd = o3d.geometry.PointCloud() # 创建一个空的点云对象
pcd.points = o3d.utility.Vector3dVector(points_data[:, :3]) # 从NumPy数组points_data中分配3D点（取前3列作为 x,y,z 坐标）
pcd.colors = o3d.utility.Vector3dVector(colors_data)  # 分配颜色信息
o3d.visualization.draw_geometries([pcd]) # 可视化点云


# # 创建alpha shape并进行mesh化
# alpha = 0.015  # 设置alpha值
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
#
# # 可视化网格化后的结果
# o3d.visualization.draw_geometries([mesh], window_name="网格化结果")
