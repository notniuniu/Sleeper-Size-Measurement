import numpy as np
import open3d as o3d
import time
import os
from typing import Dict, Tuple, List, Optional


def identify_bottom_left_region(pcd: o3d.geometry.PointCloud, box_ratios: Tuple[float, float, float] = (0.3, 0.3, 0.3)) -> Tuple[np.ndarray, np.ndarray]:
    """
    识别点云边界盒左下角区域
    
    Args:
        pcd: 输入点云
        box_ratios: 左下角区域在x、y、z三个轴上分别占整个点云边界盒的比例
        
    Returns:
        bottom_left_points: 左下角区域的点
        mask: 左下角区域点的掩码
    """
    points = np.asarray(pcd.points)
    
    # 计算点云边界盒
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    
    # 确保box_ratios是三元组
    if isinstance(box_ratios, (int, float)):
        box_ratios = (box_ratios, box_ratios, box_ratios)
    
    # 计算左下角区域的边界（三个轴独立设置）
    ranges = max_bound - min_bound
    bottom_left_max = min_bound + np.array(box_ratios) * ranges
    
    # 筛选左下角区域的点
    mask = np.all(points <= bottom_left_max, axis=1)
    bottom_left_points = points[mask]
    
    return bottom_left_points, mask


def cluster_points(points: np.ndarray, eps: float = 0.02, min_points: int = 10) -> Tuple[List[np.ndarray], List[int]]:
    """
    对点云进行DBSCAN聚类
    
    Args:
        points: 输入点云数据
        eps: 邻域半径
        min_points: 形成聚类的最小点数
        
    Returns:
        clusters: 聚类后的点集合列表
        labels: 每个点的聚类标签
    """
    if len(points) == 0:
        return [], np.array([])
    
    # 创建临时点云进行聚类
    temp_pcd = o3d.geometry.PointCloud()
    temp_pcd.points = o3d.utility.Vector3dVector(points)
    
    # 执行DBSCAN聚类
    labels = np.array(temp_pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    
    # 整理聚类结果
    clusters = []
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        if label != -1:  # 排除噪声点
            cluster_points = points[labels == label]
            clusters.append(cluster_points)
    
    return clusters, labels


def calculate_center_region(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    计算点云的几何中心
    
    Args:
        pcd: 输入点云
        
    Returns:
        center: 点云几何中心坐标
    """
    points = np.asarray(pcd.points)
    center = np.mean(points, axis=0)
    return center


def filter_clusters_by_center_distance(clusters: List[np.ndarray], center: np.ndarray) -> List[np.ndarray]:
    """
    根据聚类中心与点云几何中心的距离筛选聚类结果
    
    Args:
        clusters: 聚类后的点集合列表
        center: 点云几何中心坐标
        
    Returns:
        filtered_clusters: 筛选后的聚类结果
    """
    if not clusters:
        return []
    
    # 计算每个聚类的中心
    cluster_centers = [np.mean(cluster, axis=0) for cluster in clusters]
    
    # 计算每个聚类中心到点云中心的距离
    distances = [np.linalg.norm(cluster_center - center) for cluster_center in cluster_centers]
    
    # 如果只有一个聚类，直接返回
    if len(distances) == 1:
        return clusters
    
    # 找出距离最小的聚类
    min_distance_idx = np.argmin(distances)
    
    # 保留距离中心最近的聚类
    filtered_clusters = [clusters[min_distance_idx]]
    
    return filtered_clusters


def remove_bottom_left_noise(pcd: o3d.geometry.PointCloud, 
                           box_ratios: Tuple[float, float, float] = (0.3, 0.3, 0.3), 
                           cluster_eps: float = 0.02, 
                           min_cluster_points: int = 10, 
                           adaptive_params: bool = True, 
                           verbose: bool = False) -> o3d.geometry.PointCloud:
    """
    去除点云边界盒左下角区域的噪声点
    
    Args:
        pcd: 输入点云（无自带法向量信息）
        box_ratio: 左下角区域占整个点云边界盒的比例
        cluster_eps: DBSCAN聚类的邻域半径
        min_cluster_points: 形成聚类的最小点数
        adaptive_params: 是否根据点云规模自适应调整参数
        verbose: 是否显示详细信息
        
    Returns:
        denoised_pcd: 去除噪声后的点云
    """
    start_time = time.time()
    
    # 验证输入
    if not isinstance(pcd, o3d.geometry.PointCloud) or len(pcd.points) == 0:
        if verbose:
            print("无效的点云输入")
        return o3d.geometry.PointCloud()
    
    points = np.asarray(pcd.points)
    total_points = len(points)
    
    if verbose:
        print(f"原始点云点数: {total_points}")
    
    # 自适应调整参数
    if adaptive_params:
        # 根据点云规模调整参数
        if total_points > 100000:
            cluster_eps = max(cluster_eps, 0.05)
            min_cluster_points = max(min_cluster_points, 20)
        elif total_points < 10000:
            cluster_eps = min(cluster_eps, 0.01)
            min_cluster_points = max(5, min_cluster_points)
        
        # 根据点云密度调整聚类参数
        points_range = points.max(axis=0) - points.min(axis=0)
        avg_range = np.mean(points_range)
        if avg_range > 10:  # 大规模点云
            cluster_eps *= 2
    
    # 1. 识别左下角区域
    bottom_left_points, bottom_left_mask = identify_bottom_left_region(pcd, box_ratios)
    bottom_left_count = len(bottom_left_points)
    
    if verbose:
        print(f"左下角区域点数: {bottom_left_count} ({bottom_left_count/total_points*100:.2f}%)")
    
    if bottom_left_count == 0:
        if verbose:
            print("未检测到左下角区域点，返回原始点云")
        return pcd
    
    # 2. 对左下角区域进行聚类
    clusters, labels = cluster_points(bottom_left_points, eps=cluster_eps, min_points=min_cluster_points)
    
    if verbose:
        print(f"左下角区域聚类结果: {len(clusters)} 个有效聚类")
    
    if not clusters:
        # 如果没有有效聚类，移除整个左下角区域
        if verbose:
            print("左下角区域无有效聚类，移除整个区域")
        filtered_points = points[~bottom_left_mask]
    else:
        # 3. 计算整个点云的几何中心
        center = calculate_center_region(pcd)
        
        # 4. 根据距离中心的远近筛选聚类结果
        filtered_clusters = filter_clusters_by_center_distance(clusters, center)
        
        if verbose:
            print(f"筛选后保留聚类数: {len(filtered_clusters)}")
        
        # 5. 整合处理后的点云数据
        # 合并保留的聚类点
        if filtered_clusters:
           保留的左下角点 = np.vstack(filtered_clusters)
        else:
            保留的左下角点 = np.array([])
        
        # 获取非左下角区域的点
        non_bottom_left_points = points[~bottom_left_mask]
        
        # 合并保留的点
        if 保留的左下角点.size > 0:
            filtered_points = np.vstack([non_bottom_left_points, 保留的左下角点])
        else:
            filtered_points = non_bottom_left_points
    
    # 创建去噪后的点云
    denoised_pcd = o3d.geometry.PointCloud()
    denoised_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    
    # 如果原始点云有颜色信息，保留颜色
    if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
        colors = np.asarray(pcd.colors)
        if 'filtered_clusters' in locals() and filtered_clusters:
            # 获取保留的左下角点的颜色
            bottom_left_colors = colors[bottom_left_mask]
            保留的左下角颜色 = []
            
            # 恢复每个聚类点的颜色
            current_idx = 0
            for cluster in filtered_clusters:
                cluster_size = len(cluster)
                保留的左下角颜色.append(bottom_left_colors[current_idx:current_idx+cluster_size])
                current_idx += cluster_size
            
            if 保留的左下角颜色:
                保留的左下角颜色 = np.vstack(保留的左下角颜色)
                # 获取非左下角区域的颜色
                non_bottom_left_colors = colors[~bottom_left_mask]
                
                # 合并颜色
                if 保留的左下角颜色.size > 0:
                    filtered_colors = np.vstack([non_bottom_left_colors, 保留的左下角颜色])
                else:
                    filtered_colors = non_bottom_left_colors
                
                denoised_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        else:
            # 如果移除整个左下角区域，只保留非左下角区域的颜色
            filtered_colors = colors[~bottom_left_mask]
            denoised_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
    
    denoised_points_count = len(denoised_pcd.points)
    removed_points = total_points - denoised_points_count
    
    if verbose:
        print(f"去噪后点数: {denoised_points_count}")
        print(f"移除点数: {removed_points} ({removed_points/total_points*100:.2f}%)")
        print(f"处理时间: {time.time() - start_time:.2f}秒")
    
    return denoised_pcd


def process_point_cloud_with_bottom_left_denoising(input_pcd: o3d.geometry.PointCloud, 
                                                  params: Optional[Dict] = None) -> Dict:
    """
    完整的点云边界盒左下角去噪处理流程
    
    Args:
        input_pcd: 输入点云
        params: 去噪参数，包含以下键：
            - box_ratio: 左下角区域比例
            - cluster_eps: 聚类邻域半径
            - min_cluster_points: 最小聚类点数
            - adaptive_params: 是否自适应参数
            - verbose: 是否显示详细信息
        
    Returns:
        result: 处理结果字典
    """
    # 设置默认参数
    default_params = {
        'box_ratios': (0.3, 0.3, 0.3),  # x、y、z三个轴的比例
        'cluster_eps': 0.02,
        'min_cluster_points': 10,
        'adaptive_params': True,
        'verbose': True
    }
    
    # 更新参数
    if params is None:
        params = default_params
    else:
        for key, value in default_params.items():
            if key not in params:
                params[key] = value
    
    start_time = time.time()
    
    # 执行去噪
    denoised_pcd = remove_bottom_left_noise(input_pcd, **params)
    
    # 计算统计信息
    original_points = len(input_pcd.points)
    denoised_points = len(denoised_pcd.points)
    removal_rate = (original_points - denoised_points) / original_points * 100 if original_points > 0 else 0
    
    result = {
        'denoised_pcd': denoised_pcd,
        'original_points': original_points,
        'denoised_points': denoised_points,
        'removed_points': original_points - denoised_points,
        'removal_rate': removal_rate,
        'processing_time': time.time() - start_time,
        'parameters_used': params
    }
    
    if params['verbose']:
        print("\n=== 点云边界盒左下角去噪处理完成 ===")
        print(f"原始点数: {original_points}")
        print(f"去噪后点数: {denoised_points}")
        print(f"移除点数: {result['removed_points']}")
        print(f"移除率: {removal_rate:.2f}%")
        print(f"处理时间: {result['processing_time']:.2f}秒")
    
    return result


if __name__ == "__main__":
    # 测试代码
    print("点云边界盒左下角去噪工具测试")
    print("============================")
    
    # 测试文件列表（按优先级排序）
    test_files = [
        "results/03预旋转.ply",
    ]
    
    # 查找可用的测试文件
    input_file = None
    for file_path in test_files:
        if os.path.exists(file_path):
            input_file = file_path
            print(f"找到测试文件: {input_file}")
            break
    
   
    # 加载测试文件
    try:
        print(f"正在加载点云文件: {input_file}")
        pcd = o3d.io.read_point_cloud(input_file)
        print(f"成功加载点云，包含{len(pcd.points)}个点")
    except Exception as e:
        print(f"加载点云失败: {str(e)}")
        # 创建默认测试点云
        print("创建默认测试点云...")
        points = np.random.rand(1000, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    
    # 定义测试参数
    test_params = {
        'box_ratios': (0.01, 0.25, 0.3),  # x、y、z三个轴的比例
        'cluster_eps': 1,                   # 聚类邻域半径
        'min_cluster_points': 10,           # 最小聚类点数
        'adaptive_params': True, # 自适应参数
        'verbose': True          # 显示详细信息
    }
    
    print("\n开始执行边界盒左下角去噪处理...")
    print(f"使用参数: {test_params}")
    
    try:
        # 执行去噪处理
        result = process_point_cloud_with_bottom_left_denoising(pcd, test_params)
        denoised_pcd = result['denoised_pcd']
        
        # 保存去噪结果
        output_file = 'denoised_bottom_left_result.ply'
        save_result = o3d.io.write_point_cloud(output_file, denoised_pcd)
        print(f"\n去噪结果保存{'成功' if save_result else '失败'}: {output_file}")
        
        # 显示处理统计信息
        print("\n=== 处理统计 ===")
        print(f"原始点数: {result['original_points']}")
        print(f"去噪后点数: {result['denoised_points']}")
        print(f"移除点数: {result['removed_points']}")
        print(f"移除率: {result['removal_rate']:.2f}%")
        print(f"处理时间: {result['processing_time']:.2f}秒")
        
        # 提供可视化选项
        print("\n是否需要可视化结果？可以使用以下代码:")
        print("  o3d.visualization.draw_geometries([denoised_pcd], window_name='去噪后点云')")
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
