from point_preprocess import load_and_process_points
from point_preprocess import save_points_to_ply
from point_preprocess import extract_largest_cluster
from point_preprocess import adaptive_downsample
import numpy as np

points = load_and_process_points("C:\\images\\1218\\pointData-0-2512130112.txt")
downsampled_output = "C:\\images\\1218\\pointData-0-2512130112-0下采样.ply"
cluster_output = "C:\\images\\1218\\pointData-0-2512130112-0聚类去噪.ply"
    
# 检查第一个点云是否为空
if points.size == 0:
    print("警告：第一个点云数据为空，无法继续处理")


downsampled_pcd = adaptive_downsample(points)

print(f"降采样后的点云点数: {len(downsampled_pcd.points)}")
print(f"降采样率: {(1 - len(downsampled_pcd.points) / len(points)) * 100:.2f}%")
# 3. 基于距离的聚类算法
print("\n=== 执行基于距离的聚类算法 ===")
largest_cluster_pcd = extract_largest_cluster(downsampled_pcd)

# 4. 保存降采样后的点云
or_points_np = np.asarray(points)
or_result = save_points_to_ply(or_points_np, "C:\\Users\\30583\\Desktop\\202509-轨枕点云检测项目\\数据测试采集历史文件\\results(1)\\1106上午测试数据\\f30ed67b-07e7-456d-b417-b5bf03568f35-0原始.ply", binary=True)
print(f"原始点云文件保存{'成功' if or_result else '失败'}: C:\\Users\\30583\\Desktop\\202509-轨枕点云检测项目\\数据测试采集历史文件\\results(1)\\1106上午测试数据\\f30ed67b-07e7-456d-b417-b5bf03568f35-1原始.ply")

print("\n保存降采样后的点云到PLY文件...")
downsampled_points_np = np.asarray(downsampled_pcd.points)
downsampled_result = save_points_to_ply(downsampled_points_np, downsampled_output, binary=True)
print(f"降采样点云文件保存{'成功' if downsampled_result else '失败'}: {downsampled_output}")

# 5. 保存最大聚类的点云
if len(largest_cluster_pcd.points) > 0:
    print("\n保存最大聚类点云到PLY文件...")
    largest_cluster_points_np = np.asarray(largest_cluster_pcd.points)
    cluster_result = save_points_to_ply(largest_cluster_points_np, cluster_output, binary=True)
    print(f"最大聚类点云文件保存{'成功' if cluster_result else '失败'}: {cluster_output}")

print("\n=== 点云处理流程完成 ===")
