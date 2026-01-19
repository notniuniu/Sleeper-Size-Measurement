import open3d as o3d
import numpy as np
import struct
import os
from point_bottom_left_denoising import process_point_cloud_with_bottom_left_denoising
from point_preprocess import extract_largest_cluster
from PCA_Pose_Calibration import advanced_pca_with_reference


class PointCloudSegmenter:
    """点云分割模块 - 用于双块式轨枕点云分割"""
    
    def __init__(self, output_dir="measurements_results/segmentation_results"):
        """
        初始化点云分割器

        Args:
            output_dir (str): 输出数据目录
        """
        self.output_dir = output_dir
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def _preprocess_point_cloud(self, pcd):
        """
        预处理点云：下采样（无需去噪）
        
        Args:
            pcd (o3d.geometry.PointCloud): 原始点云
            
        Returns:
            o3d.geometry.PointCloud: 预处理后的点云
        """
        # 创建新的点云对象进行复制
        processed_pcd = o3d.geometry.PointCloud()
        processed_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        
        print(f"预处理前点云点数: {len(processed_pcd.points)}")
        
        # 自适应下采样 - 调整参数确保保留足够点数量
        points = np.asarray(processed_pcd.points)
        # 计算点云包围盒
        min_bound = points.min(axis=0)
        max_bound = points.max(axis=0)
        diagonal_length = np.linalg.norm(max_bound - min_bound)
        
        # 根据点云规模调整体素大小，确保保留足够的点
        num_points = len(points)
        if num_points > 100000:
            voxel_size = diagonal_length * 0.001  # 减小体素大小，保留更多点
        elif num_points > 50000:
            voxel_size = diagonal_length * 0.002
        else:
            # 对于点数较少的点云，使用更小的体素或不进行下采样
            voxel_size = diagonal_length * 0.003
        
        # 执行下采样
        if voxel_size > 0:
            processed_pcd_down = processed_pcd.voxel_down_sample(voxel_size=voxel_size)
            
            # 确保下采样后点数量不低于最低阈值
            min_points_required = 5000  # 设置最低点数量要求
            if len(processed_pcd_down.points) < min_points_required and len(processed_pcd.points) >= min_points_required:
                # 如果下采样后点太少且原始点足够多，使用更小的体素大小
                voxel_size = diagonal_length * 0.002
                processed_pcd_down = processed_pcd.voxel_down_sample(voxel_size=voxel_size)
                print(f"调整下采样参数，体素大小: {voxel_size:.4f}")
            
            # 如果调整后仍然太少，保留原始点云
            if len(processed_pcd_down.points) < min_points_required and len(processed_pcd.points) >= min_points_required:
                print("下采样后点数量不足，保留原始点云")
            else:
                processed_pcd = processed_pcd_down
            
            print(f"下采样后点云点数: {len(processed_pcd.points)}, 体素大小: {voxel_size:.4f}")
        
        return processed_pcd
    
    def _align_point_cloud(self, pcd):
        """
        使用PCA对点云进行对齐，确保点云的长轴（主成分）与X轴完全平行
        
        Args:
            pcd (o3d.geometry.PointCloud): 输入点云
            
        Returns:
            o3d.geometry.PointCloud: 对齐后的点云，长轴与X轴平行
        """
        import numpy as np
        import logging
        
        # 获取logger
        logger = logging.getLogger(__name__)
        
        points = np.asarray(pcd.points)
        
        # 检查点云是否为空
        if len(points) == 0:
            logger.warning("输入点云为空，无法进行对齐")
            return pcd
        
        # 计算协方差矩阵
        centered_points = points - np.mean(points, axis=0)
        cov_matrix = np.cov(centered_points, rowvar=False)
        
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 排序特征向量（从大到小），确保第一主成分对应最大特征值
        idx = eigenvalues.argsort()[::-1]
        sorted_eigenvalues = eigenvalues[idx]
        sorted_eigenvectors = eigenvectors[:, idx]
        
        # 现在我们需要确保第一主成分（长轴）与X轴平行
        # 获取第一主成分（长轴方向向量）
        major_axis = sorted_eigenvectors[:, 0]
        
        # 我们需要构建一个旋转矩阵，将长轴旋转到与X轴平行
        # 首先计算从当前长轴到X轴的旋转
        x_axis = np.array([1, 0, 0])
        
        # 计算旋转轴（两个向量的叉积）
        rotation_axis = np.cross(major_axis, x_axis)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        # 计算旋转角度（使用点积）
        cos_theta = np.dot(major_axis, x_axis)
        # 防止数值误差导致的范围问题
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        # 如果旋转轴接近零向量（两个向量已经接近平行），则不需要旋转
        if rotation_axis_norm < 1e-6:
            # 检查方向，如果方向相反则绕Y轴旋转180度
            if cos_theta < 0:
                # 绕Y轴旋转180度
                rot_matrix = np.array([
                    [-1, 0, 0],
                    [0, 1, 0],
                    [0, 0, -1]
                ])
            else:
                rot_matrix = np.eye(3)
        else:
            # 归一化旋转轴
            rotation_axis = rotation_axis / rotation_axis_norm
            
            # 使用罗德里格斯旋转公式构建旋转矩阵
            I = np.eye(3)
            K = np.array([
                [0, -rotation_axis[2], rotation_axis[1]],
                [rotation_axis[2], 0, -rotation_axis[0]],
                [-rotation_axis[1], rotation_axis[0], 0]
            ])
            
            rot_matrix = I + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        
        # 首先应用PCA旋转将主成分对齐
        pca_aligned = np.dot(centered_points, sorted_eigenvectors)
        
        # 然后应用额外的旋转，确保长轴与X轴完全平行
        fully_aligned = np.dot(pca_aligned, rot_matrix)
        
        # 创建新的点云对象
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(fully_aligned)
        
        # 保留原始点云的颜色（如果有）
        if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
            aligned_pcd.colors = pcd.colors
        
        logger.info("点云对齐完成，长轴已与X轴完全平行")
        return aligned_pcd

    def remove_outliers_with_dbscan(self, pcd, eps=20, min_points=8, min_points_for_clustering=10, verbose=True):
        """
        使用DBSCAN聚类去除点云中的离群点，保留最大点云簇
        
        Args:
            pcd (o3d.geometry.PointCloud): 输入点云
            eps (float): 第一次DBSCAN聚类的邻域半径
            min_points (int): 第一次DBSCAN聚类的最小样本数
            looser_eps (float): 第二次尝试的邻域半径（更宽松）
            looser_min_points (int): 第二次尝试的最小样本数（更宽松）
            min_points_for_clustering (int): 执行聚类所需的最小点数
            verbose (bool): 是否打印详细日志
            
        Returns:
            o3d.geometry.PointCloud: 处理后的点云
        """
        points = np.asarray(pcd.points)
        
        # 如果点数足够，尝试使用DBSCAN聚类
        if len(points) > min_points_for_clustering:
            try:
                if verbose:
                    print(f"应用DBSCAN聚类，参数: eps={eps}, min_points={min_points}")
                
                labels = o3d.geometry.PointCloud.cluster_dbscan(pcd, eps=eps, min_points=min_points, print_progress=False)
                
                # 找出所有聚类的大小
                cluster_sizes = {}
                noise_count = 0
                
                for label in labels:
                    if label != -1:  # -1表示噪声点
                        cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
                    else:
                        noise_count += 1
                
                if verbose:
                    print(f"聚类分析: 检测到 {len(cluster_sizes)} 个聚类，噪声点数量: {noise_count}")
                    # 显示各聚类大小详情
                    if cluster_sizes:
                        for label, size in cluster_sizes.items():
                            print(f"聚类 {label}: {size} 个点")
                
                # 如果找到聚类，选择最大的聚类
                if cluster_sizes:
                    largest_cluster_label = max(cluster_sizes, key=cluster_sizes.get)
                    largest_size = cluster_sizes[largest_cluster_label]
                    
                    if verbose:
                        print(f"选择最大聚类 {largest_cluster_label}，包含 {largest_size} 个点")
                    
                    # 只保留最大聚类中的点
                    largest_cluster_indices = [i for i, label in enumerate(labels) if label == largest_cluster_label]
                    filtered_pcd = pcd.select_by_index(largest_cluster_indices)
                    
                    if verbose:
                        print(f"成功保留最大点云簇，点数: {len(filtered_pcd.points)}, 移除点数: {len(points) - len(filtered_pcd.points)}")
                    
                    return filtered_pcd
            except Exception as e:
                # # 聚类过程中出错，回退到统计离群点检测
                # if verbose:
                #     print(f"聚类处理出错: {e}，回退到统计离群点检测")
                # filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                return filtered_pcd
        else:
            # 点数太少，直接使用统计离群点检测
            if verbose:
                print(f"点数太少({len(points)}点)，直接使用统计离群点检测")
            filtered_pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            return filtered_pcd
    
    def _segment_by_height(self, pcd):
        """
        基于高度进行初步分割，并使用平面拟合优化接触面识别
        先识别接触平面并旋转整体点云使其与XY平面平行，再进行分割
        
        Args:
            pcd (o3d.geometry.PointCloud): 输入点云
            
        Returns:
            tuple: (上层点云-承轨台区域, 下层点云-复合单元块区域)
        """
        points = np.asarray(pcd.points)
        
        # 计算Z值范围
        z_values = points[:, 2]
        min_z = np.min(z_values)
        max_z = np.max(z_values)
        z_range = max_z - min_z
        
        print(f"原始Z值范围: {min_z:.4f} - {max_z:.4f}, 跨度: {z_range:.4f}")
        
        # ============ 第一步：初步识别平面并旋转整体点云 ============
        # 使用动态阈值进行初步分割，仅用于识别平面
        initial_z_threshold = min_z + 0.6 * z_range
        initial_upper_mask = z_values > initial_z_threshold
        upper_points_initial = points[initial_upper_mask]
        
        # 用于存储旋转后的点云
        rotated_pcd = pcd
        
        if len(upper_points_initial) > 0:
            # 计算上层点云的Z值范围
            upper_z_min = np.min(upper_points_initial[:, 2])
            upper_z_max = np.max(upper_points_initial[:, 2])
            upper_z_range = upper_z_max - upper_z_min
            
            # 提取上层点云的底部区域用于平面拟合
            bottom_percentage = 0.4
            bottom_threshold = upper_z_min + bottom_percentage * upper_z_range
            upper_bottom_mask = upper_points_initial[:, 2] <= bottom_threshold
            upper_bottom_points = upper_points_initial[upper_bottom_mask]
            
            print(f"用于平面拟合的点数: {len(upper_bottom_points)}")
            
            if len(upper_bottom_points) > 100:
                # 创建临时点云对象用于平面拟合
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(upper_bottom_points)
                
                try:
                    # 使用RANSAC算法拟合平面
                    plane_model, inliers = temp_pcd.segment_plane(
                        distance_threshold=z_range * 0.01,
                        ransac_n=3,
                        num_iterations=1000
                    )
                    
                    a, b, c, d = plane_model
                    print(f"拟合平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
                    print(f"平面内点数量: {len(inliers)}")
                    
                    # 旋转整体点云使平面与XY平面平行
                    rotated_pcd = self._rotate_pcd_to_align_plane(pcd, plane_model)
                    print("整体点云已旋转校正")
                    
                except Exception as e:
                    print(f"平面拟合失败: {e}，使用原始点云继续")
            else:
                print("平面拟合点数不足，使用原始点云继续")
        else:
            print("初步上层点云为空，使用原始点云继续")
        
        # ============ 第二步：在旋转后的点云上进行正式分割 ============
        points = np.asarray(rotated_pcd.points)
        z_values = points[:, 2]
        min_z = np.min(z_values)
        max_z = np.max(z_values)
        z_range = max_z - min_z
        
        print(f"旋转后Z值范围: {min_z:.4f} - {max_z:.4f}, 跨度: {z_range:.4f}")
        
        # 使用动态阈值进行分割
        initial_z_threshold = min_z + 0.6 * z_range
        initial_upper_mask = z_values > initial_z_threshold

        # 创建分割后的点云
        upper_pcd = self._extract_segment(rotated_pcd, initial_upper_mask)
        
        # 从上层点云中提取底部区域用于精细平面拟合
        upper_points = points[initial_upper_mask]
        if len(upper_points) > 0:
            upper_z_min = np.min(upper_points[:, 2])
            upper_z_max = np.max(upper_points[:, 2])
            upper_z_range = upper_z_max - upper_z_min
            
            bottom_percentage = 0.4
            bottom_threshold = upper_z_min + bottom_percentage * upper_z_range
            upper_bottom_mask = upper_points[:, 2] <= bottom_threshold
            upper_bottom_points = upper_points[upper_bottom_mask]
            
            print(f"上层底部区域点数: {len(upper_bottom_points)}, 占上层点云的比例: {len(upper_bottom_points)/len(upper_points)*100:.1f}%")
            
            if len(upper_bottom_points) > 100:
                temp_pcd = o3d.geometry.PointCloud()
                temp_pcd.points = o3d.utility.Vector3dVector(upper_bottom_points)
                
                try:
                    # 在旋转后的点云上再次拟合平面（应该接近水平）
                    plane_model, inliers = temp_pcd.segment_plane(
                        distance_threshold=z_range * 0.01,
                        ransac_n=3,
                        num_iterations=1000
                    )
                    
                    a, b, c, d = plane_model
                    print(f"旋转后拟合平面方程: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")
                    print(f"平面内点数量: {len(inliers)}")
                    
                    denominator = np.sqrt(a*a + b*b + c*c)
                    if denominator > 0:
                        # 计算每个点到平面的有符号距离
                        signed_distances = (a*points[:,0] + b*points[:,1] + c*points[:,2] + d) / denominator
                        
                        # 定义接触平面附近的阈值
                        contact_threshold = z_range * 0.005
                        
                        # 重新分类
                        contact_mask = np.abs(signed_distances) < contact_threshold
                        above_plane_mask = signed_distances > contact_threshold
                        below_plane_mask = signed_distances < -contact_threshold
                        
                        final_upper_mask = above_plane_mask
                        final_lower_mask = contact_mask | below_plane_mask
                        
                        # 确保所有点都被分类
                        all_classified = np.all(final_upper_mask | final_lower_mask)
                        if not all_classified:
                            unclassified_mask = ~(final_upper_mask | final_lower_mask)
                            final_upper_mask = final_upper_mask | (unclassified_mask & initial_upper_mask)
                            final_lower_mask = final_lower_mask | (unclassified_mask & ~initial_upper_mask)
                        
                        print(f"平面拟合优化完成：上层点云点数: {np.sum(final_upper_mask)}, 下层点云点数: {np.sum(final_lower_mask)}")
                        
                        upper_mask = final_upper_mask
                        lower_mask = final_lower_mask
                    else:
                        upper_mask = initial_upper_mask
                        lower_mask = ~initial_upper_mask
                    
                except Exception as e:
                    print(f"平面拟合失败: {e}，使用初始分割结果")
                    upper_mask = initial_upper_mask
                    lower_mask = ~initial_upper_mask
            else:
                print("上层底部区域点数量不足，使用初始分割结果")
                upper_mask = initial_upper_mask
                lower_mask = ~initial_upper_mask
        else:
            print("上层点云为空，使用初始分割结果")
            upper_mask = initial_upper_mask
            lower_mask = ~initial_upper_mask
        
        # 创建分割后的点云
        upper_pcd = self._extract_segment(rotated_pcd, upper_mask)
        lower_pcd = self._extract_segment(rotated_pcd, lower_mask)
        
        # 去除离群点
        upper_pcd = self.remove_outliers_with_dbscan(upper_pcd)
        
        # 从旋转后的原始点云中减去upper_pcd得到lower_pcd
        if len(rotated_pcd.points) > 0 and len(upper_pcd.points) > 0:
            try:
                kdtree = o3d.geometry.KDTreeFlann(rotated_pcd)
                upper_indices = []
                
                for point in upper_pcd.points:
                    [_, idx, _] = kdtree.search_knn_vector_3d(point, 1)
                    upper_indices.append(idx[0])
                
                all_indices = np.arange(len(rotated_pcd.points))
                lower_indices = np.setdiff1d(all_indices, np.array(upper_indices))
                lower_pcd = rotated_pcd.select_by_index(lower_indices)
                
                print(f"通过点云减法创建lower_pcd，点数: {len(lower_pcd.points)}")
                    
            except Exception as e:
                print(f"点云减法出错: {e}，保留原有的lower_pcd")
                lower_pcd, _ = lower_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        print(f"承轨台点数: {len(upper_pcd.points)}")
        print(f"单元块点数: {len(lower_pcd.points)}")
        
        return upper_pcd, lower_pcd
    
    def _rotate_pcd_to_align_plane(self, pcd, plane_model):
        """
        根据拟合的平面旋转整体点云，使平面与XY平面严格平行
        
        Args:
            pcd (o3d.geometry.PointCloud): 输入点云
            plane_model (list): 平面方程参数 [a, b, c, d]
            
        Returns:
            o3d.geometry.PointCloud: 旋转后的点云
        """
        a, b, c, d = plane_model
        
        # 计算平面法向量（归一化）
        plane_normal = np.array([a, b, c])
        plane_normal = plane_normal / np.linalg.norm(plane_normal)
        
        # 确保法向量指向Z正方向
        if plane_normal[2] < 0:
            plane_normal = -plane_normal
        
        # 目标法向量
        target_normal = np.array([0.0, 0.0, 1.0])
        
        # 检查是否已经平行
        dot_product = np.dot(plane_normal, target_normal)
        if np.abs(dot_product - 1.0) < 1e-6:
            print("平面已与XY平面平行，无需旋转")
            return pcd
        
        # 计算旋转轴和角度
        rotation_axis = np.cross(plane_normal, target_normal)
        rotation_axis_norm = np.linalg.norm(rotation_axis)
        
        if rotation_axis_norm < 1e-6:
            if dot_product < 0:
                rotation_axis = np.array([1.0, 0.0, 0.0])
                rotation_angle = np.pi
            else:
                return pcd
        else:
            rotation_axis = rotation_axis / rotation_axis_norm
            rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        
        print(f"平面法向量: [{plane_normal[0]:.4f}, {plane_normal[1]:.4f}, {plane_normal[2]:.4f}]")
        print(f"旋转轴: [{rotation_axis[0]:.4f}, {rotation_axis[1]:.4f}, {rotation_axis[2]:.4f}]")
        print(f"旋转角度: {np.degrees(rotation_angle):.4f}°")
        
        # 构建旋转矩阵
        rotation_matrix = self._rodrigues_rotation_matrix(rotation_axis, rotation_angle)
        
        # 以点云中心为旋转中心
        points = np.asarray(pcd.points)
        rotation_center = np.mean(points, axis=0)
        
        # 应用旋转
        return self._apply_rotation(pcd, rotation_matrix, rotation_center)
    
    def _rodrigues_rotation_matrix(self, axis, angle):
        """
        使用Rodrigues公式计算旋转矩阵
        
        Args:
            axis (np.ndarray): 归一化的旋转轴
            angle (float): 旋转角度（弧度）
            
        Returns:
            np.ndarray: 3x3旋转矩阵
        """
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        return R
    
    def _apply_rotation(self, pcd, rotation_matrix, center):
        """
        将旋转矩阵应用到点云，绕指定中心旋转
        
        Args:
            pcd (o3d.geometry.PointCloud): 输入点云
            rotation_matrix (np.ndarray): 3x3旋转矩阵
            center (np.ndarray): 旋转中心
            
        Returns:
            o3d.geometry.PointCloud: 旋转后的点云
        """
        rotated_pcd = o3d.geometry.PointCloud()
        
        points = np.asarray(pcd.points)
        centered_points = points - center
        rotated_points = np.dot(centered_points, rotation_matrix.T)
        final_points = rotated_points + center
        
        rotated_pcd.points = o3d.utility.Vector3dVector(final_points)
        
        if pcd.has_colors():
            rotated_pcd.colors = pcd.colors
        
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
            rotated_normals = np.dot(normals, rotation_matrix.T)
            rotated_pcd.normals = o3d.utility.Vector3dVector(rotated_normals)
        
        return rotated_pcd
    

    def _extract_segment(self, pcd, mask):
        """
        根据掩码提取点云子集
        
        Args:
            pcd (o3d.geometry.PointCloud): 原始点云
            mask (np.ndarray): 布尔掩码数组
            
        Returns:
            o3d.geometry.PointCloud: 提取的点云子集
        """
        segment = o3d.geometry.PointCloud()
        points = np.asarray(pcd.points)[mask]
        segment.points = o3d.utility.Vector3dVector(points)
        
        # 如果原始点云有颜色，也提取颜色
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)[mask]
            segment.colors = o3d.utility.Vector3dVector(colors)
        
        # 如果原始点云有法线，也提取法线
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)[mask]
            segment.normals = o3d.utility.Vector3dVector(normals)
        
        return segment
    
    def _save_segment(self, pcd, file_name):
        """
        保存分割后的点云片段
        
        Args:
            pcd (o3d.geometry.PointCloud): 点云数据
            file_name (str): 输出文件名
        """
        output_path = os.path.join(self.output_dir, file_name)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"分割结果已保存至: {output_path}, 点数: {len(pcd.points)}")
    
    def _segment_by_x_axis(self, pcd, ratios=[1815, 1000, 1815], remove_outliers=True, save_temp=False):
        """
        基于X轴对输入点云进行比例分割，返回左、中、右三部分点云
        
        参数:
            pcd: 输入点云
            ratios: X轴方向分割比例，默认为[1815, 1031, 1815]
            remove_outliers: 是否去除离群点
            save_temp: 是否保存临时结果
            
        返回:
            left_pcd, middle_pcd, right_pcd: 分割后的左、中、右三部分点云
        """
        # 去除离群点（如果需要）
        if remove_outliers:
            pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            print(f"去除离群点后的点数: {len(pcd.points)}")
        
        # 获取点云X轴范围
        points = np.asarray(pcd.points)
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        x_range = max_x - min_x
        
        # 计算分割阈值
        total_ratio = sum(ratios)
        left_ratio, middle_ratio, right_ratio = ratios
        
        # 计算分割边界
        left_bound = min_x + (left_ratio / total_ratio) * x_range
        middle_bound = left_bound + (middle_ratio / total_ratio) * x_range
        
        # 创建分割掩码
        left_mask = points[:, 0] <= left_bound
        middle_mask = (points[:, 0] > left_bound) & (points[:, 0] <= middle_bound)
        right_mask = points[:, 0] > middle_bound
        
        # 提取分割后的点云
        left_pcd = self._extract_segment(pcd, left_mask)
        middle_pcd = self._extract_segment(pcd, middle_mask)
        right_pcd = self._extract_segment(pcd, right_mask)
        
        print(f"左部点云点数: {len(left_pcd.points)}, 中部点云点数: {len(middle_pcd.points)}, 右部点云点数: {len(right_pcd.points)}")
        
        # 保存临时结果
        if save_temp and hasattr(self, 'output_dir'):
            temp_dir = os.path.join(self.output_dir, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            o3d.io.write_point_cloud(os.path.join(temp_dir, 'temp_x_left_pcd.ply'), left_pcd)
            o3d.io.write_point_cloud(os.path.join(temp_dir, 'temp_x_middle_pcd.ply'), middle_pcd)
            o3d.io.write_point_cloud(os.path.join(temp_dir, 'temp_x_right_pcd.ply'), right_pcd)
            print(f"X轴分割结果已保存至: {temp_dir}")
        
        return left_pcd, middle_pcd, right_pcd
    
    def segment(self, pcd, align=True):
        """
        执行完整的点云分割流程
        
        Args:
            pcd (o3d.geometry.PointCloud): 输入点云
            align (bool): 是否对点云进行对齐
            
        Returns:
            dict: 包含所有分割区域点云的字典
        """
        print(f"开始点云分割，原始点云点数: {len(pcd.points)}")
        
        # 1. 预处理点云
        processed_pcd = self._preprocess_point_cloud(pcd)
        
        # 2. 点云对齐（可选）
        if align:
            processed_pcd = self._align_point_cloud(processed_pcd)
        
        # 2.5 绕Z轴旋转90度，使X轴与Y轴互换
        theta = np.radians(40)  # 45度转换为弧度
        rotation_matrix0 = np.array([
            [np.cos(theta), 0, np.sin(theta)],    # 第一行：X' = cosθ*X + 0*Y + sinθ*Z
            [0, 1, 0],                           # 第二行：Y' = 0*X + 1*Y + 0*Z
            [-np.sin(theta), 0, np.cos(theta)]   # 第三行：Z' = -sinθ*X + 0*Y + cosθ*Z
        ])
        # 应用旋转矩阵
        processed_pcd.points = o3d.utility.Vector3dVector(np.dot(np.asarray(processed_pcd.points), rotation_matrix0.T))
        # 如果点云有法线信息，也需要旋转法线
        if processed_pcd.has_normals():
            processed_pcd.normals = o3d.utility.Vector3dVector(np.dot(np.asarray(processed_pcd.normals), rotation_matrix0.T))
        
        rotation_matrix1 = np.array([
            [0, -1, 0],  # 第一行：X' = 0*X - 1*Y + 0*Z
            [1, 0, 0],   # 第二行：Y' = 1*X + 0*Y + 0*Z
            [0, 0, 1]    # 第三行：Z' = 0*X + 0*Y + 1*Z
        ])
        # 应用旋转矩阵
        processed_pcd.points = o3d.utility.Vector3dVector(np.dot(np.asarray(processed_pcd.points), rotation_matrix1.T))
        # 如果点云有法线信息，也需要旋转法线
        if processed_pcd.has_normals():
            processed_pcd.normals = o3d.utility.Vector3dVector(np.dot(np.asarray(processed_pcd.normals), rotation_matrix1.T))

        theta2 = np.arctan(16.938 / 470.688)
        # 创建绕x轴顺时针旋转的旋转矩阵
        rotation_matrix2 = np.array([
            [1, 0, 0],
            [0, np.cos(theta2), np.sin(theta2)],
            [0, -np.sin(theta2), np.cos(theta2)]
        ]) 
        # 应用旋转矩阵
        processed_pcd.points = o3d.utility.Vector3dVector(np.dot(np.asarray(processed_pcd.points), rotation_matrix2.T))
        # 如果点云有法线信息，也需要旋转法线
        if processed_pcd.has_normals():
            processed_pcd.normals = o3d.utility.Vector3dVector(np.dot(np.asarray(processed_pcd.normals), rotation_matrix2.T))
        
        theta3 = np.radians(1.44936)  # 使用numpy的弧度转换函数
        # 创建绕X轴逆时针旋转的旋转矩阵
        rotation_matrix3 = np.array([
            [1, 0, 0],
            [0, np.cos(theta3), -np.sin(theta3)],
            [0, np.sin(theta3), np.cos(theta3)]
        ])
                # 应用旋转矩阵
        processed_pcd.points = o3d.utility.Vector3dVector(np.dot(np.asarray(processed_pcd.points), rotation_matrix3.T))
        # 如果点云有法线信息，也需要旋转法线
        if processed_pcd.has_normals():
            processed_pcd.normals = o3d.utility.Vector3dVector(np.dot(np.asarray(processed_pcd.normals), rotation_matrix3.T))

        aligned_pcd, _, _ = advanced_pca_with_reference(
            processed_pcd, output_pcd_path = os.path.join(self.output_dir, '04精配准.ply')
        )
        
        # 执行点云裁剪操作，保留Z坐标在-370.3至84.5区间内的点
        print("执行点云裁剪操作，保留Z坐标在-370.3至84.5区间内的点")
        
        # 提取点云的所有点
        points = np.asarray(aligned_pcd.points)
        
        # 创建掩码，保留z值在[-370.3, 84.5]范围内的点
        z_min = -370.3
        z_max = 85
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
        
        # 计算裁剪前后的点数
        original_point_count = len(points)
        filtered_point_count = np.sum(mask)
        
        # 应用掩码，获取裁剪后的点云数据
        filtered_points = points[mask]
        
        # 创建裁剪后的新点云对象
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        # 如果原始点云有法线信息，也需要对应裁剪法线数据
        if aligned_pcd.has_normals():
            normals = np.asarray(aligned_pcd.normals)
            filtered_normals = normals[mask]
            filtered_pcd.normals = o3d.utility.Vector3dVector(filtered_normals)
        
        # 如果原始点云有颜色信息，也需要对应裁剪颜色数据
        if aligned_pcd.has_colors():
            colors = np.asarray(aligned_pcd.colors)
            filtered_colors = colors[mask]
            filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        # 如果原始点云有强度信息（如果存在），也需要对应裁剪
        if hasattr(aligned_pcd, 'intensities') and aligned_pcd.intensities:
            intensities = np.asarray(aligned_pcd.intensities)
            filtered_intensities = intensities[mask]
            filtered_pcd.intensities = o3d.utility.VectorXdVector(filtered_intensities)
        
        # 更新aligned_pcd为裁剪后的点云
        aligned_pcd = filtered_pcd
        
        # 打印裁剪结果
        print(f"裁剪前点数: {original_point_count}")
        print(f"裁剪后点数: {filtered_point_count}")
        print(f"裁剪完成，保留点数: {len(aligned_pcd.points)}")
        print(f"裁剪保留比例: {(filtered_point_count / original_point_count * 100):.2f}%")
        # 保存裁剪后的点云
        cropped_pcd_path = os.path.join(self.output_dir, '04_精配准_裁剪.ply')
        o3d.io.write_point_cloud(cropped_pcd_path, aligned_pcd)
        print(f"裁剪后的点云已保存至: {cropped_pcd_path}")
        
        # test_params = {
        # 'box_ratios': (0.01, 0.25, 0.3),  # x、y、z三个轴的比例
        # 'cluster_eps': 1,                   # 聚类邻域半径
        # 'min_cluster_points': 10,           # 最小聚类点数
        # 'adaptive_params': True, # 自适应参数
        # 'verbose': True          # 显示详细信息
        # }
        # # 2.5 基于边界盒左下角去噪
        # result1 = process_point_cloud_with_bottom_left_denoising(aligned_pcd, test_params)
        # denoised_pcd = result1['denoised_pcd']

         # 2.6 基于X轴的初步分割
        left_pcd, middle_pcd, right_pcd = self._segment_by_x_axis(aligned_pcd, save_temp=False)
        
        # 3. 基于高度的初步分割
        left_upper_pcd, left_lower_pcd = self._segment_by_height(left_pcd)
        # left_lower_pcd = extract_largest_cluster(left_lower_pcd)

        right_upper_pcd, right_lower_pcd = self._segment_by_height(right_pcd)
        # right_lower_pcd = extract_largest_cluster(right_lower_pcd)
        
        
        # 处理中间区域点云
        middle_upper_pcd = self._process_middle_pcd(middle_pcd, self.output_dir)
        
        # 保存各部分点云
        o3d.io.write_point_cloud(os.path.join(self.output_dir, 'cgt_l.ply'), left_upper_pcd)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, 'dyk_l.ply'), left_lower_pcd)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, 'cgt_r.ply'), right_upper_pcd)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, 'dyk_r.ply'), right_lower_pcd)
        o3d.io.write_point_cloud(os.path.join(self.output_dir, '03预旋转.ply'), processed_pcd)
        # o3d.io.write_point_cloud(os.path.join(self.output_dir, '05去除边缘泡沫.ply'), denoised_pcd)
        
        # 构建分割结果字典
        segments = {
            'cgt_l': left_upper_pcd,
            'dyk_l': left_lower_pcd,
            'cgt_r': right_upper_pcd,
            'dyk_r': right_lower_pcd
        }
        
        # 如果中间区域点云处理成功，添加到结果字典中
        if middle_upper_pcd is not None:
            o3d.io.write_point_cloud(os.path.join(self.output_dir, 'cgt_m.ply'), middle_pcd)
            segments['cgt_m'] = middle_upper_pcd
        
        print(f"点云分割完成，共分割出 {len(segments)} 个区域")
        return segments

        
    def _process_middle_pcd(self, middle_pcd, output_dir):
        """
        处理中间区域点云，通过聚类算法分为两个子集，保留Z值较大的部分并去噪
        
        参数:
            middle_pcd: 中间区域点云
            output_dir: 输出目录
            
        返回:
            处理后的点云
        """
        if len(middle_pcd.points) == 0:
            print("中间区域点云为空，跳过处理")
            return None
        
        try:
            # 使用DBSCAN聚类算法将点云分为多个子集
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
                labels = np.array(middle_pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=False))
            
            # 获取有效的聚类标签（排除噪声点标签-1）
            unique_labels = np.unique(labels)
            valid_labels = unique_labels[unique_labels >= 0]
            
            if len(valid_labels) >= 2:
                # 计算每个聚类的Z值最大值
                points = np.asarray(middle_pcd.points)
                cluster_z_values = []
                
                for label in valid_labels:
                    cluster_points = points[labels == label]
                    if len(cluster_points) > 0:
                        max_z = np.max(cluster_points[:, 2])
                        cluster_z_values.append((label, max_z, cluster_points))
                
                # 按Z值降序排序，选择Z值较大的聚类
                cluster_z_values.sort(key=lambda x: x[1], reverse=True)
                selected_label = cluster_z_values[0][0]  # 选择Z值最大的聚类
                
                # 提取选中的点云子集
                selected_mask = labels == selected_label
                selected_pcd = self._extract_segment(middle_pcd, selected_mask)
                print(f"选择Z值较大的聚类，点数: {len(selected_pcd.points)}")
            else:
                # 如果聚类数量不足，使用Z值中值进行简单分割
                points = np.asarray(middle_pcd.points)
                z_median = np.median(points[:, 2])
                # 选择Z值大于中值的点作为目标点云
                selected_mask = points[:, 2] > z_median
                selected_pcd = self._extract_segment(middle_pcd, selected_mask)
                print(f"聚类数量不足，使用Z值中值分割，选择点数: {len(selected_pcd.points)}")
            
            # 对选中的点云进行去噪操作
            # denoised_pcd, _ = selected_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            # print(f"去噪后的点数: {len(denoised_pcd.points)}")
            
            # 保存结果为cgt_m.ply
            output_path = os.path.join(output_dir, 'cgt_m.ply')
            o3d.io.write_point_cloud(output_path, selected_pcd)
            print(f"中间区域处理结果已保存至: {output_path}")
            
            return selected_pcd
        except Exception as e:
            print(f"处理中间区域点云时出错: {str(e)}")
            return None

             

# 独立测试部分
if __name__ == "__main__":
    try:
        # 测试数据路径
        test_file = "measurements_results/02聚类去噪.ply"  # 只使用文件名，不包含目录
        output_dir = "measurements_results/segmentation_results"
        
        # 1. 读取点云数据（手动处理编码问题，支持多种文件格式）
        print(f"读取点云数据: {test_file}")
        
        # 确定文件路径
        file_path = None
        if os.path.exists(test_file):
            file_path = test_file
        else:
            data_path = os.path.join("data", test_file)
            if os.path.exists(data_path):
                file_path = data_path
            else:
                raise FileNotFoundError(f"点云文件不存在: {test_file} 或 {data_path}")
        
        # 检查文件扩展名，区分处理ply和txt文件
        _, file_ext = os.path.splitext(file_path)
        
        if file_ext.lower() == '.ply':
            # PLY文件直接使用Open3D读取
            try:
                print(f"检测到PLY文件，使用Open3D直接读取...")
                pcd = o3d.io.read_point_cloud(file_path)
                print(f"成功读取PLY文件，点数: {len(pcd.points)}")
            except Exception as e:
                print(f"读取PLY文件失败: {e}")
                raise
        elif file_ext.lower() == '.txt':
            # 使用二进制方式读取TXT文件，实现分块读取以提高效率
            points = []
            brightness_values = []
            skipped_records = 0
            
            try:
                # 定义二进制记录格式和大小
                record_format = '3fi'  # 3个float, 1个int
                record_size = struct.calcsize(record_format)
                chunk_size = 10000  # 每次读取的记录数
                
                file_size = os.path.getsize(file_path)
                total_records = file_size // record_size
                
                print(f"检测到TXT文件，使用二进制模式读取")
                print(f"文件大小: {file_size} 字节, 预计记录数: {total_records}")
                
                with open(file_path, 'rb') as file:
                    records_read = 0
                    
                    while True:
                        # 读取一个块的数据
                        chunk_data = file.read(chunk_size * record_size)
                        if not chunk_data:
                            break
                        
                        # 确保读取完整记录
                        remaining = len(chunk_data) % record_size
                        if remaining > 0:
                            chunk_data = chunk_data[:-remaining]  # 去除不完整的记录
                            file.seek(-remaining, 1)  # 回退文件指针
                        
                        if not chunk_data:
                            break
                        
                        # 解析当前块的数据
                        num_records = len(chunk_data) // record_size
                        for i in range(num_records):
                            start = i * record_size
                            end = start + record_size
                            record_data = chunk_data[start:end]
                            
                            try:
                                x, y, z, brightness = struct.unpack(record_format, record_data)
                                
                                # 检查Z值是否为NaN，如果是则跳过该记录
                                if np.isnan(z):
                                    skipped_records += 1
                                    continue
                                
                                points.append([x, y, z])
                                brightness_values.append(brightness)
                            except:
                                skipped_records += 1
                        
                        records_read += num_records
                        print(f"已处理 {records_read}/{total_records} 个记录", end='\r')
                
                print(f"\n成功读取 {len(points)} 个点")
                print(f"解析完成，共跳过 {skipped_records} 条无效数据")
            except Exception as e:
                print(f"读取文件时出错: {e}")

            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            
            print(f"成功创建点云对象，点数: {len(pcd.points)}")
        else:
            raise ValueError(f"不支持的文件格式: {file_ext}，仅支持.ply和.txt文件")
        
        print(f"原始点云点数: {len(pcd.points)}")
        
        # 2. 初始化分割器
        print("初始化点云分割器...")
        segmenter = PointCloudSegmenter(output_dir=output_dir)
        
        # 3. 执行分割（不进行点云对齐）
        print("执行点云分割...")
        segments = segmenter.segment(pcd, align=False)
        
        # 4. 显示分割结果统计
        print("\n分割结果统计:")
        for name, segment in segments.items():
            print(f"  {name}: {len(segment.points)} 点")
            
    except ImportError as e:
        print(f"导入错误: {str(e)}")
        print("请确保所有必要的模块已安装，特别是open3d和numpy")
    except FileNotFoundError as e:
        print(f"文件错误: {str(e)}")
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()