import open3d as o3d
import numpy as np


# points_data = np.loadtxt("E:/vsproject/勋仪交接/标克艾芬达暖水管/测量主管尺寸/vp/11pnts.txt", delimiter=",", dtype=np.float32)
points_data = np.loadtxt("E:/vsproject/勋仪交接/标克艾芬达暖水管/测量主管尺寸/vpc/11pnts.txt", delimiter=",")
print('shape: ', points_data.shape)
print('data type: ', points_data.dtype)

points_data[:,2] = points_data[:,2] * 36 # 将所有行的第三列乘以 21
points_data[:,0] = points_data[:,0] * 4

# points_data = np.random.rand(100, 3)  # 示例数据：100 个随机 3D 点
colors_data = np.random.rand(100, 3)  # 100 个随机 RGB 颜色数组

pcd = o3d.geometry.PointCloud() # 创建一个空的点云对象
pcd.points = o3d.utility.Vector3dVector(points_data[:, :3]) # 从NumPy数组points_data中分配3D点（取前3列作为 x,y,z 坐标）
pcd.colors = o3d.utility.Vector3dVector(colors_data)  # 分配颜色信息

# 1. 下采样（降采样）
down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
print("下采样后的点云数量:", len(np.asarray(down_pcd.points)))

# 2. 去除离群点
# 统计离群点移除
cl, ind = down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
statistical_outlier_removed = down_pcd.select_by_index(ind)
print("统计离群点移除后的点云数量:", len(np.asarray(statistical_outlier_removed.points)))

# 半径离群点移除
# cl, ind = statistical_outlier_removed.remove_radius_outlier(nb_points=16, radius=0.05)
cl, ind = statistical_outlier_removed.remove_radius_outlier(nb_points=16, radius=0.05)
radius_outlier_removed = statistical_outlier_removed.select_by_index(ind)
print("半径离群点移除后的点云数量:", len(np.asarray(radius_outlier_removed.points)))


# 3. 滤波（平滑） - 自定义均值滤波
def mean_filter(pcd, k=5):
    ''' 对每个点找到其k个最近的邻居并计算均值作为滤波后的点坐标 '''
    points = np.asarray(pcd.points)
    filtered_points = np.zeros_like(points)
    for i in range(len(points)):
        # 找到k个最近的邻居（这里简化处理，实际应用中可以使用KD树等高效方法）
        distances = np.linalg.norm(points - points[i], axis=1)
        nearest_indices = np.argsort(distances)[:k]
        filtered_points[i] = np.mean(points[nearest_indices], axis=0)
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_pcd


filtered_pcd = mean_filter(radius_outlier_removed, k=5)
print("均值滤波后的点云数量:", len(np.asarray(filtered_pcd.points)))

# 可视化原始点云和处理后的点云
# o3d.visualization.draw_geometries([pcd]) # 可视化点云
o3d.visualization.draw_geometries([pcd], window_name="原始点云")
o3d.visualization.draw_geometries([down_pcd], window_name="下采样后的点云")
o3d.visualization.draw_geometries([statistical_outlier_removed], window_name="统计离群点移除后的点云")
o3d.visualization.draw_geometries([radius_outlier_removed], window_name="半径离群点移除后的点云")
o3d.visualization.draw_geometries([filtered_pcd], window_name="均值滤波后的点云")



# # 创建alpha shape并进行mesh化
# alpha = 0.015  # 设置alpha值
# mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
#
# # 可视化网格化后的结果
# o3d.visualization.draw_geometries([mesh], window_name="网格化结果")

def visual_3pipe(): # 可视化方管，D型管，散热片缺陷
    space = 2500
    pnts_r = np.loadtxt("E:/vsproject/勋仪交接/标克艾芬达暖水管/测量主管尺寸/vpr/11pnts.txt", delimiter=",")
    pnts_r[:,2] = pnts_r[:,2] * 36 # 将所有行的第三列乘以 21
    pnts_r[:,1] = pnts_r[:,1] + space
    pnts_r[:,0] = pnts_r[:,0] * 4
    pnts_d = np.loadtxt("E:/vsproject/勋仪交接/标克艾芬达暖水管/测量主管尺寸/vpd/11pnts.txt", delimiter=",")
    pnts_d[:,2] = pnts_d[:,2] * 36 # 将所有行的第三列乘以 21
    pnts_d[:,1] = pnts_d[:,1] + space*2
    pnts_d[:,0] = pnts_d[:,0] * 2
    pnts_s = np.loadtxt("E:/vsproject/勋仪交接/标克艾芬达暖水管/测量主管尺寸/散热片划痕/11pnts.txt", delimiter=",")
    pnts_s[:,2] = pnts_s[:,2] * 16 # 将所有行的第三列乘以 21
    pnts_s[:,1] = pnts_s[:,1] + space*3
    pnts_s[:,0] = pnts_s[:,0] * 2

    def get_pcd(points_data):
        pcd = o3d.geometry.PointCloud()  # 创建一个空的点云对象
        pcd.points = o3d.utility.Vector3dVector(points_data[:, :3])  # 从NumPy数组points_data中分配3D点（取前3列作为 x,y,z 坐标）
        pcd.colors = o3d.utility.Vector3dVector(colors_data)  # 分配颜色信息
        return pcd

    pcd_r = get_pcd(pnts_r)
    pcd_d = get_pcd(pnts_d)
    pcd_s = get_pcd(pnts_s)
    o3d.visualization.draw_geometries([pcd_r,pcd_d,pcd_s]) # 可视化点云

# visual_3pipe()





