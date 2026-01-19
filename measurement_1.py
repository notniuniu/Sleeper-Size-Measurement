import open3d as o3d
import numpy as np
import os
import logging
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor
import time
from parallelogram_measurement import calculate_parallelogram_length_width
from point_preprocess import extract_largest_cluster

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('measurement_1')

class RailMeasurement:
    """双块式轨枕几何参数测量类"""
    
    def __init__(self, output_dir="test_results"):
        """
        初始化测量类
        
        Args:
            output_dir (str): 中间结果输出目录
        """
        self.output_dir = output_dir
        # 确保输出目录存在（包括所有父目录）
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")
        
        # 测量参数配置
        self.min_points_threshold = 20  # 最小点数阈值
        self.normalization_factor = 1  # 毫米转换因子
    
    def _copy_point_cloud(self, pcd):
        """复制点云对象"""
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if hasattr(pcd, 'colors') and len(pcd.colors) > 0:
            new_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
        return new_pcd
    
    def _remove_outliers(self, pcd, nb_neighbors=20, std_ratio=2.0):
        """移除统计离群点"""
        if len(pcd.points) < self.min_points_threshold:
            logger.warning("点云点数过少，跳过离群点移除")
            return pcd
        
        try:
            _, inliers = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            filtered_pcd = pcd.select_by_index(inliers)
            logger.info(f"离群点移除: 从 {len(pcd.points)} 点到 {len(filtered_pcd.points)} 点")
            return filtered_pcd
        except Exception as e:
            logger.error(f"离群点移除失败: {e}")
            return pcd
    
    def _get_pca_direction(self, points):
        """使用PCA获取点云的主方向"""
        if len(points) < 3:
            logger.error("点太少，无法计算PCA")
            return None
        
        try:
            # 计算PCA
            centered_points = points - np.mean(points, axis=0)
            cov_matrix = np.cov(centered_points, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            
            # 排序特征向量（从大到小）
            idx = eigenvalues.argsort()[::-1]
            sorted_eigenvectors = eigenvectors[:, idx]
            
            return sorted_eigenvectors
        except Exception as e:
            logger.error(f"PCA计算失败: {e}")
            return None
    
    def _fit_plane_ransac(self, pcd, distance_threshold=None, ransac_n=3, num_iterations=1000):
        """使用RANSAC拟合平面"""
        if len(pcd.points) < self.min_points_threshold:
            logger.warning("点云点数过少，跳过平面拟合")
            return None, None
        
        try:
            # 如果未提供距离阈值，基于点云规模自动计算
            if distance_threshold is None:
                points = np.asarray(pcd.points)
                min_bound = points.min(axis=0)
                max_bound = points.max(axis=0)
                diagonal_length = np.linalg.norm(max_bound - min_bound)
                distance_threshold = diagonal_length * 0.001  # 动态阈值
            
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=ransac_n,
                num_iterations=num_iterations
            )
            
            logger.info(f"平面拟合成功: {len(inliers)}/{len(pcd.points)} 点在平面内")
            return plane_model, inliers
        except Exception as e:
            logger.error(f"平面拟合失败: {e}")
            return None, None
    
    def _extract_percentile_points(self, pcd, axis=2, lower_percentile=None, upper_percentile=None):
        """提取特定百分位数范围的点"""
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return o3d.geometry.PointCloud()
        
        values = points[:, axis]
        result_points = []
        
        if lower_percentile is not None and upper_percentile is not None:
            lower_bound = np.percentile(values, lower_percentile)
            upper_bound = np.percentile(values, upper_percentile)
            result_points = points[(values >= lower_bound) & (values <= upper_bound)]
        elif lower_percentile is not None:
            lower_bound = np.percentile(values, lower_percentile)
            result_points = points[values >= lower_bound]
        elif upper_percentile is not None:
            upper_bound = np.percentile(values, upper_percentile)
            result_points = points[values <= upper_bound]
        
        result_pcd = o3d.geometry.PointCloud()
        result_pcd.points = o3d.utility.Vector3dVector(result_points)
        logger.info(f"提取{axis}轴方向特定百分位数点: {len(result_points)} 点")
        return result_pcd
    
    def _fit_line(self, points, direction='x'):
        """
        拟合直线（平行于X轴或Y轴）
        
        Args:
            points: 点云数据数组
            direction: 拟合方向，'x'表示平行于Y轴（x=常数），'y'表示平行于X轴（y=常数）
            
        Returns:
            dict: 拟合结果，包含类型和值
        """
        if len(points) < 2:
            logger.error("点太少，无法拟合直线")
            return None
        
        try:
            if direction == 'x':
                # 拟合平行于Y轴的直线：x = 常数
                avg_x = np.mean(points[:, 0])
                return {'type': 'x', 'value': avg_x}
            elif direction == 'y':
                # 拟合平行于X轴的直线：y = 常数
                avg_y = np.mean(points[:, 1])
                return {'type': 'y', 'value': avg_y}
        except Exception as e:
            logger.error(f"直线拟合失败: {e}")
            return None
    
    def _find_edge_points_by_binning(self, pcd, axis='y', edge_type='min', bins=10):
        """通过分箱找到边缘点"""
        points = np.asarray(pcd.points)
        if len(points) == 0:
            return np.array([])
        
        try:
            # 获取分箱轴的值
            bin_values = points[:, 0] if axis == 'x' else points[:, 1]
            target_values = points[:, 1] if axis == 'x' else points[:, 0]
            
            # 创建分箱
            min_val, max_val = np.min(bin_values), np.max(bin_values)
            bin_edges = np.linspace(min_val, max_val, bins + 1)
            
            edge_points = []
            for i in range(bins):
                # 找到当前分箱内的点
                bin_mask = (bin_values >= bin_edges[i]) & (bin_values < bin_edges[i+1])
                bin_points = target_values[bin_mask]
                
                if len(bin_points) > 0:
                    # 找到极值点
                    if edge_type == 'min':
                        extreme_idx = np.argmin(bin_points)
                    else:  # 'max'
                        extreme_idx = np.argmax(bin_points)
                    
                    # 获取对应的三维点
                    bin_points_3d = points[bin_mask]
                    edge_points.append(bin_points_3d[extreme_idx])
            
            logger.info(f"分箱边缘检测: 找到 {len(edge_points)} 个边缘点")
            return np.array(edge_points)
        except Exception as e:
            logger.error(f"边缘点检测失败: {e}")
            return np.array([])
    
    def _save_point_cloud(self, pcd, filename, color=None):
        """保存点云到文件"""
        if len(pcd.points) == 0:
            logger.warning(f"空点云，跳过保存: {filename}")
            return False
        
        try:
            save_pcd = self._copy_point_cloud(pcd)
            if color is not None:
                save_pcd.paint_uniform_color(color)
            
            filepath = os.path.join(self.output_dir, filename)
            o3d.io.write_point_cloud(filepath, save_pcd)
            logger.info(f"保存点云: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存点云失败: {e}")
            return False
    
    def _filter_outliers(self, points, percentile=95):
        """
        过滤点云中的异常值点
        
        Args:
            points: 点云数据数组
            percentile: 百分位数阈值，用于过滤远离主要分布的点
            
        Returns:
            numpy.ndarray: 过滤后的点云数据
        """
        if len(points) == 0:
            return points
            
        try:
            # 计算每个点的Y坐标（假设我们关心的是Y方向的异常值）
            y_coords = points[:, 1]
            
            # 计算四分位数和四分位距
            q1 = np.percentile(y_coords, 25)
            q3 = np.percentile(y_coords, 75)
            iqr = q3 - q1
            
            # 计算边界
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 过滤异常值
            mask = (y_coords >= lower_bound) & (y_coords <= upper_bound)
            filtered_points = points[mask]
            
            # 如果过滤后点太少，使用另一种方法
            if len(filtered_points) < len(points) * 0.5 or len(filtered_points) < 3:
                # 使用百分位数过滤更宽松的阈值
                lower_percentile = max(0, 50 - percentile/2)
                upper_percentile = min(100, 50 + percentile/2)
                lower_bound = np.percentile(y_coords, lower_percentile)
                upper_bound = np.percentile(y_coords, upper_percentile)
                mask = (y_coords >= lower_bound) & (y_coords <= upper_bound)
                filtered_points = points[mask]
            
            logger.info(f"过滤异常值: {len(points)} -> {len(filtered_points)} 点")
            return filtered_points
        except Exception as e:
            logger.error(f"异常值过滤失败: {e}")
            return points
    
    def _save_line_points(self, line_params, point_cloud, filename, color=[1, 0, 0]):
        """保存直线表示的点云"""
        if line_params is None:
            return False
        
        try:
            points = np.asarray(point_cloud.points)
            if len(points) == 0:
                return False
            
            # 创建表示直线的点
            if line_params['type'] == 'x':
                # x = constant，沿Y轴延伸
                y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
                z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
                y_vals = np.linspace(y_min, y_max, 50)
                line_points = np.array([[line_params['value'], y, (z_min+z_max)/2] for y in y_vals])
            else:  # 'y'
                # y = constant，沿X轴延伸
                x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
                z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])
                x_vals = np.linspace(x_min, x_max, 50)
                line_points = np.array([[x, line_params['value'], (z_min+z_max)/2] for x in x_vals])
            
            # 创建点云并保存
            line_pcd = o3d.geometry.PointCloud()
            line_pcd.points = o3d.utility.Vector3dVector(line_points)
            line_pcd.paint_uniform_color(color)
            
            return self._save_point_cloud(line_pcd, filename)
        except Exception as e:
            logger.error(f"保存直线失败: {e}")
            return False
    
    def calculate_symmetry(self, cgt, dyk, test_mode=False, is_right_side=True):
        """
        计算对称度s1
        
        Args:
            cgt: 承轨台点云
            dyk: 单元块点云
            test_mode: 是否保存中间结果
            
        Returns:
            dict: 包含对称度值和相关信息
        """
        start_time = time.time()
        logger.info("开始计算对称度s1")
        
        result = {
            'value': 0.0,
            'success': False,
            'symmetry_axis': None,
            'top_edge': None,
            'bottom_edge': None
        }
        
        # 步骤1: 找到承轨台与X轴平行的对称轴
        cgt_top = self._extract_percentile_points(cgt, axis=2, lower_percentile=50, upper_percentile=80)
        # o3d.visualization.draw_geometries([cgt_top], window_name="临时可视化 - cgt_top点云")
        
        # 记录原始点云点数
        original_point_count = len(cgt_top.points)
        # 调用extract_largest_cluster函数
        filtered_cgt_top = extract_largest_cluster(cgt_top)
        # 计算去除的点的百分比
        filtered_point_count = len(filtered_cgt_top.points)
        removed_percentage = (original_point_count - filtered_point_count) / original_point_count * 100
        
        # 如果去除的点超过20%，则保留原来的cgt_top
        if removed_percentage > 20:
            logger.info(f"去除了{removed_percentage:.1f}%的点，超过20%，保留原始点云")
        else:
            cgt_top = filtered_cgt_top
            logger.info(f"去除了{removed_percentage:.1f}%的点，使用过滤后的点云")

        # 步骤1.5: 按X轴和Y轴范围过滤点云
        # 获取点云坐标
        cgt_top_points = np.asarray(cgt_top.points)
        
        # 定义过滤范围
        if is_right_side:
            x_min, x_max = -833.51, -484.97
            y_min, y_max = -149.19, 51.06
        else:
            x_min, x_max = -2357.05, 1986.75
            y_min, y_max = -149.19, 51.06
        
        # 创建过滤掩码
        x_mask = (cgt_top_points[:, 0] >= x_min) & (cgt_top_points[:, 0] <= x_max)
        y_mask = (cgt_top_points[:, 1] >= y_min) & (cgt_top_points[:, 1] <= y_max)
        combined_mask = x_mask & y_mask
        
        # 应用过滤
        filtered_points = cgt_top_points[combined_mask]
        
        # 记录过滤统计
        original_count = len(cgt_top_points)
        filtered_count = len(filtered_points)
        filter_percentage = (original_count - filtered_count) / original_count * 100
        
        # 创建新的点云对象
        if filtered_count > 0:
            cgt_top = o3d.geometry.PointCloud()
            cgt_top.points = o3d.utility.Vector3dVector(filtered_points)
            logger.info(f"坐标范围过滤: 原始{original_count}点 -> 过滤后{filtered_count}点，去除{filter_percentage:.1f}%的点")
            logger.info(f"X轴范围: {x_min} 到 {x_max}, Y轴范围: {y_min} 到 {y_max}")
        else:
            logger.warning("坐标范围过滤后没有点，保留原始点云")
        
        
        # 临时可视化cgt_top点云
        # o3d.visualization.draw_geometries([cgt_top], window_name="临时可视化 - cgt_top点云")


        cgt_points = np.asarray(cgt_top.points)

        
        # 确定对称轴（平行于X轴）
        # 使用更稳健的方法计算对称轴位置
        # 1. 计算Y坐标的统计信息
        y_coords = cgt_points[:, 1]
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        y_range = y_max - y_min
        
        # 3. 也可以考虑使用Y方向边界框的中点
        y_midpoint = (y_min + y_max) / 2
        
        # 4. 选择最终的对称轴位置：优先使用中位数（更稳健），但可以根据需要调整权重
        centroid_y = y_midpoint 
        result['symmetry_axis'] = centroid_y

        
        
        if test_mode:
            # 保存对称轴点云
            x_min, x_max = np.min(cgt_points[:, 0]), np.max(cgt_points[:, 0])
            z_min, z_max = np.min(cgt_points[:, 2]), np.max(cgt_points[:, 2])
            axis_points = np.array([[x, centroid_y, (z_min+z_max)/2] 
                                  for x in np.linspace(x_min, x_max, 50)])



            axis_pcd = o3d.geometry.PointCloud()
            axis_pcd.points = o3d.utility.Vector3dVector(axis_points)

            self._save_point_cloud(axis_pcd, "对称度_承轨台对称轴.ply", [1, 0, 0])
        
        # 步骤2: 处理单元块点云
        dyk_filtered = self._remove_outliers(dyk)
        
        # 提取底部点云（Z值最最高的50%点）
        bottom_points = self._extract_percentile_points(dyk_filtered, axis=2, lower_percentile=40, upper_percentile=60)
        
        if len(bottom_points.points) < self.min_points_threshold:
            logger.error("单元块底部点云点数不足")
            return result
        
        if test_mode:
            self._save_point_cloud(bottom_points, "对称度_单元块底面.ply", [0, 1, 0])
        
        # 步骤3: 找到单元块底部两条边缘线
        # 找到上边线点（Y最大值）
        # 注意：这里使用axis='x'表示在X轴上进行分箱，然后在每个分箱内寻找Y坐标的极值
        top_edge_points = self._find_edge_points_by_binning(bottom_points, axis='x', edge_type='max', bins=15)
        # 找到下边线点（Y最小值）
        bottom_edge_points = self._find_edge_points_by_binning(bottom_points, axis='x', edge_type='min', bins=15)
        
        # 过滤异常值点，提高边缘点质量
        top_edge_points = self._filter_outliers(top_edge_points, percentile=95)
        bottom_edge_points = self._filter_outliers(bottom_edge_points, percentile=95)
        
        # 确保有足够的点进行拟合
        min_edge_points = 3
        if len(top_edge_points) < min_edge_points or len(bottom_edge_points) < min_edge_points:
            logger.warning(f"边缘点数量不足: 上边线{len(top_edge_points)}点, 下边线{len(bottom_edge_points)}点")
            # 如果点数量太少，尝试减少分箱数量并重试
            if len(top_edge_points) < min_edge_points:
                top_edge_points = self._find_edge_points_by_binning(bottom_points, axis='x', edge_type='max', bins=8)
            if len(bottom_edge_points) < min_edge_points:
                bottom_edge_points = self._find_edge_points_by_binning(bottom_points, axis='x', edge_type='min', bins=8)
        
        # 拟合边缘线（平行于X轴的直线，即y=常数）
        top_line = self._fit_line(top_edge_points, direction='y') if len(top_edge_points) > 0 else None
        bottom_line = self._fit_line(bottom_edge_points, direction='y') if len(bottom_edge_points) > 0 else None

        if top_line is None or bottom_line is None:
            logger.error("边缘线拟合失败")
            return result
        
        result['top_edge'] = top_line['value']
        result['bottom_edge'] = bottom_line['value']
        
        if test_mode:
            # 保存边缘点和边缘线
            top_edge_pcd = o3d.geometry.PointCloud()
            
            # 合并两个nx3数组
            combined_top_points = np.vstack((top_edge_points, axis_points))
            top_edge_pcd.points = o3d.utility.Vector3dVector(combined_top_points)

            _, S1 = calculate_parallelogram_length_width(top_edge_pcd, 
                        visualize=False,output_file =os.path.join(self.output_dir, '对称度_单元块边缘线_上.ply'))
            
            bottom_edge_pcd = o3d.geometry.PointCloud()
            # 合并两个nx3数组
            combined_bottom_points = np.vstack((bottom_edge_points, axis_points))
            bottom_edge_pcd.points = o3d.utility.Vector3dVector(combined_bottom_points)

            _, S2 = calculate_parallelogram_length_width(bottom_edge_pcd, 
                        visualize=False,output_file =os.path.join(self.output_dir, '对称度_单元块边缘线_下.ply'))
            
            symmetry_value = S1 - S2

            self._save_line_points(top_line, bottom_points, "对称度_单元块边缘线_上.ply", [0, 0, 1])
            self._save_line_points(bottom_line, bottom_points, "对称度_单元块边缘线_下.ply", [1, 1, 0])

        
        # 步骤4: 计算对称度
        # 计算在XY平面上边缘线到对称轴的垂直距离
        # 对称轴是y=centroid_y，上边线是y=top_line['value']，下边线是y=bottom_line['value']
        # 垂直距离就是Y坐标差的绝对值
        S1_line = abs(top_line['value'] - centroid_y)
        S2_line = abs(bottom_line['value'] - centroid_y)
        symmetry_value_line = S1_line - S2_line
        logger.info(f"拟合直线法_S1_line: {S1_line:.6f}, S2_line: {S2_line:.6f}, symmetry_value_line: {symmetry_value_line:.6f}")
        
        if test_mode:
            # 可视化S1和S2距离
            # 选择一个X坐标的中点
            all_points = np.concatenate([np.asarray(cgt.points), np.asarray(bottom_points.points)])
            x_mid = (np.min(all_points[:, 0]) + np.max(all_points[:, 0])) / 2
            z_mid = (np.min(all_points[:, 2]) + np.max(all_points[:, 2])) / 2
            
            # 创建S1距离的可视化线段（从对称轴到上边线）
            point1_S1 = [x_mid, centroid_y, z_mid]
            point2_S1 = [x_mid, top_line['value'], z_mid]
            S1_line_points = np.array([point1_S1, point2_S1])
            S1_line_pcd = o3d.geometry.PointCloud()
            S1_line_pcd.points = o3d.utility.Vector3dVector(S1_line_points)
            self._save_point_cloud(S1_line_pcd, "对称度_S1_距离测量线.ply", [0, 0, 1])
            
            # 创建S2距离的可视化线段（从对称轴到下边线）
            point1_S2 = [x_mid, centroid_y, z_mid]
            point2_S2 = [x_mid, bottom_line['value'], z_mid]
            S2_line_points = np.array([point1_S2, point2_S2])
            S2_line_pcd = o3d.geometry.PointCloud()
            S2_line_pcd.points = o3d.utility.Vector3dVector(S2_line_points)
            self._save_point_cloud(S2_line_pcd, "对称度_S2_距离测量线.ply", [1, 1, 0])
        
        # 转换为毫米
        symmetry_value_mm = symmetry_value * self.normalization_factor
        # if symmetry_value_mm > 10.0:
        #     symmetry_value_mm = symmetry_value_mm % 10.0
        
        result['value'] = symmetry_value_mm
        result['success'] = True
        
        logger.info(f"关键点统计法_对称度s1计算完成: {symmetry_value_mm:.2f} mm, 耗时: {time.time()-start_time:.2f}s")
        return result
    
    def calculate_height(self, cgt, dyk, test_mode=False):
        """
        计算承轨台高度h
        
        Args:
            cgt: 承轨台点云
            dyk: 单元块点云
            test_mode: 是否保存中间结果
            
        Returns:
            dict: 包含高度值和相关信息
        """
        start_time = time.time()
        logger.info("开始计算承轨台高度h")
        
        result = {
            'value': 0.0,
            'success': False,
            'top_plane': None,
            'bottom_plane': None
        }
        
        # 步骤1: 提取承轨台上表面
        # 筛选上半部分点（Z值最高的70%）
        top_surface_points = self._extract_percentile_points(cgt, axis=2, lower_percentile=30)
        
        if len(top_surface_points.points) < self.min_points_threshold:
            logger.error("承轨台表面点云点数不足")
            return result
        
        # 拟合上表面平面
        top_plane_model, top_inliers = self._fit_plane_ransac(top_surface_points)
        if top_plane_model is None:
            return result
        
        result['top_plane'] = top_plane_model
        
        # 筛选出位于平面内的点
        top_plane_points = np.asarray(top_surface_points.points)[top_inliers]
        
        # 创建只包含中间部分的点云对象
        top_plane_pcd = o3d.geometry.PointCloud()
        top_plane_pcd.points = o3d.utility.Vector3dVector(top_plane_points)
        
        if test_mode:
            self._save_point_cloud(top_plane_pcd, "承轨台高度_承轨台顶面.ply", [1, 0, 0])
        
        # 步骤2: 提取单元块底面 - 使用边界框方法
        dyk_points = np.asarray(dyk.points)
        if len(dyk_points) < 200:
            logger.error("单元块点云点数不足")
            return result
        
        # 创建单元块点云对象
        dyk_pcd = o3d.geometry.PointCloud()
        dyk_pcd.points = o3d.utility.Vector3dVector(dyk_points)
        
        # 获取单元块的边界框
        bbox = dyk_pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.min_bound
        max_bound = bbox.max_bound
        
        logger.info(f"单元块边界框: 最小值={min_bound}, 最大值={max_bound}")
        
        # 提取底面内部点 - 由于单元块是长方体，底面是Z值最小的面
        # 我们选择底面区域内的点（略高于最小Z值，避免边界噪声）
        # 在X和Y方向上，我们选择边界框内部区域（避免边缘效应）
        buffer_ratio = 0.1  # 边界缓冲区比例
        z_buffer = 0.005   # Z轴方向的缓冲区高度
        
        x_min, y_min, z_min = min_bound
        x_max, y_max, z_max = max_bound
        
        # 计算内部区域的边界
        x_internal_min = x_min + (x_max - x_min) * buffer_ratio
        x_internal_max = x_max - (x_max - x_min) * buffer_ratio
        y_internal_min = y_min + (y_max - y_min) * buffer_ratio
        y_internal_max = y_max - (y_max - y_min) * buffer_ratio
        z_internal_max = z_min + z_buffer
        
        # 筛选底面内部点
        bottom_interior_mask = (dyk_points[:, 0] >= x_internal_min) & \
                             (dyk_points[:, 0] <= x_internal_max) & \
                             (dyk_points[:, 1] >= y_internal_min) & \
                             (dyk_points[:, 1] <= y_internal_max) & \
                             (dyk_points[:, 2] <= z_internal_max)
        
        bottom_interior_points = dyk_points[bottom_interior_mask]
        logger.info(f"通过边界框方法提取到底面内部点: {len(bottom_interior_points)}点")
        
        # 如果内部点不足，扩大筛选范围
        if len(bottom_interior_points) < 100:
            logger.warning(f"底面内部点不足，扩大筛选范围")
            # 减小缓冲区比例
            buffer_ratio = 0.05
            z_buffer = 0.01
            
            x_internal_min = x_min + (x_max - x_min) * buffer_ratio
            x_internal_max = x_max - (x_max - x_min) * buffer_ratio
            y_internal_min = y_min + (y_max - y_min) * buffer_ratio
            y_internal_max = y_max - (y_max - y_min) * buffer_ratio
            z_internal_max = z_min + z_buffer
            
            # 重新筛选
            bottom_interior_mask = (dyk_points[:, 0] >= x_internal_min) & \
                                 (dyk_points[:, 0] <= x_internal_max) & \
                                 (dyk_points[:, 1] >= y_internal_min) & \
                                 (dyk_points[:, 1] <= y_internal_max) & \
                                 (dyk_points[:, 2] <= z_internal_max)
            
            bottom_interior_points = dyk_points[bottom_interior_mask]
            logger.info(f"扩大范围后提取到底面内部点: {len(bottom_interior_points)}点")
        
        # 如果仍然点不足，回退到使用Z值最小的一部分点
        if len(bottom_interior_points) < 50:
            logger.warning(f"仍然点不足，回退到使用Z值最小的点")
            # 取Z值最小的20%的点
            num_points_to_take = max(100, int(len(dyk_points) * 0.02))
            sorted_indices = np.argsort(dyk_points[:, 2])
            bottom_interior_points = dyk_points[sorted_indices[:num_points_to_take]]
            logger.info(f"回退后提取到底部点: {len(bottom_interior_points)}点")
        
        # 使用底面内部点拟合平面
        bottom_interior_pcd = o3d.geometry.PointCloud()
        bottom_interior_pcd.points = o3d.utility.Vector3dVector(bottom_interior_points)
        
        bottom_plane_model, bottom_inliers = self._fit_plane_ransac(bottom_interior_pcd)
        if bottom_plane_model is None:
            return result
        
        result['bottom_plane'] = bottom_plane_model
        
        # 筛选出位于拟合平面内的点
        bottom_plane_points = bottom_interior_points[bottom_inliers]
        logger.info(f"成功提取底面平面内点: {len(bottom_plane_points)}点")
        
        bottom_plane_pcd = o3d.geometry.PointCloud()
        bottom_plane_pcd.points = o3d.utility.Vector3dVector(bottom_plane_points)
        
        if test_mode:
            self._save_point_cloud(bottom_plane_pcd, "承轨台高度_单元块底面.ply", [0, 1, 0])
        
        # 步骤3: 计算高度
        # 找到承轨台上表面平面的中心点
        top_centroid = np.mean(top_plane_points, axis=0)
        
        # 计算单元块底面的中心点
        bottom_centroid = np.mean(bottom_plane_points, axis=0)
        
        # 检查底面平面是否平行于XY平面
        # 平面方程: ax + by + cz + d = 0
        # 如果平行于XY平面，则a≈0, b≈0, c≈1
        a, b, c, d = bottom_plane_model
        is_parallel_to_xy = (abs(a) < 0.1 and abs(b) < 0.1 and abs(abs(c) - 1) < 0.1)
        
        if is_parallel_to_xy:
            # 如果底面平行于XY平面，使用垂直距离计算
            logger.info("底面平面平行于XY平面，使用垂直距离计算高度")
            distance = abs(a*top_centroid[0] + b*top_centroid[1] + c*top_centroid[2] + d) / np.sqrt(a*a + b*b + c*c)
            if test_mode:
                # 保存高度测量线
                # 从中心点垂直向下到底面
                normal = np.array([a, b, c])
                normal = normal / np.linalg.norm(normal)
                bottom_point = top_centroid - distance * normal
                
                height_line_points = np.array([top_centroid, bottom_point])
                height_line_pcd = o3d.geometry.PointCloud()
                height_line_pcd.points = o3d.utility.Vector3dVector(height_line_points)
                self._save_point_cloud(height_line_pcd, "承轨台高度_高度测量线.ply", [0, 0, 1])
        else:
            # 如果底面不平行于XY平面，使用Z值差计算
            logger.info("底面平面不平行于XY平面，使用Z值差计算高度")
            distance = abs(top_centroid[2] - bottom_centroid[2])
        
            # 可视化高度测量线
            if test_mode:
                # 创建从顶面中心点到底面中心点的线段
                line_points = np.array([top_centroid, bottom_centroid])
                line_pcd = o3d.geometry.PointCloud()
                line_pcd.points = o3d.utility.Vector3dVector(line_points)
                self._save_point_cloud(line_pcd, "承轨台高度_高度测量线.ply", [1, 0, 0])
        
        # 转换为毫米
        height_mm = distance * self.normalization_factor
        
        result['value'] = height_mm
        result['success'] = True
        

        
        logger.info(f"承轨台高度h计算完成: {height_mm:.2f} mm, 耗时: {time.time()-start_time:.2f}s")
        return result
    
    def calculate_side_width(self, cgt, dyk, test_mode=False, is_right_side=False):
        """
        计算侧面宽度we
        
        Args:
            cgt: 承轨台点云
            dyk: 单元块点云
            test_mode: 是否保存中间结果
            is_right_side: 是否为右侧（右侧时找X最大值）
            
        Returns:
            dict: 包含宽度值和相关信息
        """
        start_time = time.time()
        logger.info(f"开始计算侧面宽度we，{'右侧' if is_right_side else '左侧'}")
        
        result = {
            'value': 0.0,
            'success': False,
            'cgt_edge': None,
            'dyk_edge': None
        }
        
        # 步骤1: 找到承轨台边缘
        edge_type = 'max' if is_right_side else 'min'
        cgt_top = self._extract_percentile_points(cgt, axis=2, lower_percentile=20,upper_percentile=70)
        cgt_edge_points = self._find_edge_points_by_binning(cgt_top, axis='y', edge_type=edge_type,bins=100)
        cgt_edge_pcd = o3d.geometry.PointCloud()
        cgt_edge_pcd.points = o3d.utility.Vector3dVector(cgt_edge_points)
        self._save_point_cloud(cgt_edge_pcd, "测试.ply", [1, 0, 0])


        if len(cgt_edge_points) < 5:  # 至少需要几个点来拟合
            logger.error("承轨台边缘点不足")
            return result
        
        # 拟合承轨台边缘线（平行于Y轴）
        cgt_edge_line = self._fit_line(cgt_edge_points, direction='x')
        if cgt_edge_line is None:
            return result
        
        result['cgt_edge'] = cgt_edge_line['value']
        
        if test_mode:
            # 保存承轨台边缘点和线
            cgt_edge_pcd = o3d.geometry.PointCloud()
            cgt_edge_pcd.points = o3d.utility.Vector3dVector(cgt_edge_points)
            self._save_point_cloud(cgt_edge_pcd, f"侧面宽度_承轨台_{'右侧' if is_right_side else '左侧'}_边缘.ply", [1, 0, 0])
            # self._save_line_points(cgt_edge_points, cgt, f"侧面宽度_承轨台_{'右侧' if is_right_side else '左侧'}_边缘线.ply", [1, 0, 0])
        
        # 步骤2: 找到单元块左侧边缘
        # 提取底部点云
        dyk_bottom = self._extract_percentile_points(dyk, axis=2, upper_percentile=50)

        
        if len(dyk_bottom.points) < self.min_points_threshold:
            logger.error("单元块底部点云不足")
            return result
        
        # 直接在x轴方向上查找最edge_type的五个点
        points = np.asarray(dyk_bottom.points)
        
        # 检查points是否为空
        if len(points) == 0:
            logger.error("单元块底部点云为空")
            return result
            
        # 获取x坐标
        x_coords = points[:, 0]
        
        # 根据edge_type确定查找最大还是最小的点
        if edge_type == 'max':
            # 找到x坐标最大的5个点的索引
            indices = np.argsort(x_coords)[-500:]
        else:  # edge_type == 'min'
            # 找到x坐标最小的5个点的索引
            indices = np.argsort(x_coords)[:500]
            
        # 提取这5个点
        dyk_edge_points = points[indices]
        
        # 如果找到的点不足5个，使用所有可用点
        if len(dyk_edge_points) < 5:
            logger.warning(f"只找到{len(dyk_edge_points)}个边缘点，不足5个")
        
        print(f"直接在x轴方向找到{len(dyk_edge_points)}个{edge_type}边缘点")
        
        if len(dyk_edge_points) < 5:
            logger.error("单元块边缘点不足")
            return result
        
        # 拟合单元块边缘线（平行于Y轴）
        dyk_edge_line = self._fit_line(dyk_edge_points, direction='x')
        if dyk_edge_line is None:
            return result
                # 创建表示直线的点
        points_test = np.asarray(dyk_bottom.points)
        if dyk_edge_line['type'] == 'x':
            # x = constant，沿Y轴延伸
            y_min, y_max = np.min(points_test[:, 1]), np.max(points_test[:, 1])
            z_min, z_max = np.min(points_test[:, 2]), np.max(points_test[:, 2])
            y_vals = np.linspace(y_min, y_max, 50)
            dyk_edge_line_points = np.array([[dyk_edge_line['value'], y, (z_min+z_max)/2] for y in y_vals])
        else:  # 'y'
            # y = constant，沿X轴延伸
            x_min, x_max = np.min(points_test[:, 0]), np.max(points_test[:, 0])
            z_min, z_max = np.min(points_test[:, 2]), np.max(points_test[:, 2])
            x_vals = np.linspace(x_min, x_max, 50)
            dyk_edge_line_points = np.array([[x, dyk_edge_line['value'], (z_min+z_max)/2] for x in x_vals])
        
        # 创建点云并保存
        dyk_edge_line_pcd = o3d.geometry.PointCloud()
        dyk_edge_line_pcd.points = o3d.utility.Vector3dVector(dyk_edge_line_points)
        
        result['dyk_edge'] = dyk_edge_line['value']
        
        if test_mode:
            # 保存单元块边缘点和线
            dyk_edge_pcd = o3d.geometry.PointCloud()
            dyk_edge_pcd.points = o3d.utility.Vector3dVector(dyk_edge_points)
            # self._save_point_cloud(dyk_edge_pcd, f"侧面宽度_单元块_{'右侧' if is_right_side else '左侧'}_边缘.ply", [0, 1, 0])
            self._save_line_points(dyk_edge_pcd, dyk_bottom, f"侧面宽度_单元块_{'右侧' if is_right_side else '左侧'}_边缘线.ply", [0, 1, 0])
        

        
        # 步骤3: 计算宽度we（两条边缘线在X方向的垂直距离）
        # width = abs(cgt_edge_line['value'] - dyk_edge_line['value'])
        
        # # 转换为毫米
        # width_mm = width * self.normalization_factor
        # logger.info(f"拟合直线法_侧面宽度we计算完成: {width_mm:.2f} mm, 耗时: {time.time()-start_time:.2f}s")

    
        edge_pcd = o3d.geometry.PointCloud()
        combined_edge_points = np.vstack((cgt_edge_points, dyk_edge_line_points))
        edge_pcd.points = o3d.utility.Vector3dVector(combined_edge_points)
        width_1, width_2 = calculate_parallelogram_length_width(edge_pcd, 
                        visualize=False,output_file =os.path.join(self.output_dir, f"侧面宽度_测量线_{'右侧' if is_right_side else '左侧'}.ply"))
        
        # 从width_1和width_2中选择更接近80的值作为width_mm
        if abs(width_1 - 80) < abs(width_2 - 80):
            width_mm = width_1
        else:
            width_mm = width_2
            
        result['value'] = width_mm
        result['success'] = True
        logger.info(f"关键点统计法_侧面宽度we计算完成: {width_mm:.2f} mm, 耗时: {time.time()-start_time:.2f}s")

        if test_mode:
            # 添加宽度测量的可视化
            # 从承轨台边缘到单元块边缘创建一条垂直线段
            # 选择一个中间的Y坐标和Z坐标
            cgt_points = np.asarray(cgt.points)
            dyk_points = np.asarray(dyk.points)
            
            # 计算中间Y和Z值
            y_mid = (np.min(np.concatenate([cgt_points[:, 1], dyk_points[:, 1]])) + 
                     np.max(np.concatenate([cgt_points[:, 1], dyk_points[:, 1]]))) / 2
            z_mid = (np.min(np.concatenate([cgt_points[:, 2], dyk_points[:, 2]])) + 
                     np.max(np.concatenate([cgt_points[:, 2], dyk_points[:, 2]]))) / 2
            
            # 创建两个端点
            point1 = [cgt_edge_line['value'], y_mid, z_mid]
            point2 = [dyk_edge_line['value'], y_mid, z_mid]
            width_line_points = np.array([point1, point2])
            
            # 创建点云并保存
            width_line_pcd = o3d.geometry.PointCloud()
            width_line_pcd.points = o3d.utility.Vector3dVector(width_line_points)
            # self._save_point_cloud(width_line_pcd, f"侧面宽度_测量线_{'右侧' if is_right_side else '左侧'}.ply", [0, 0, 1])
        
        
        return result
    
    def measure_rail_components(self, cgt, dyk, test_mode=False, is_right_side=False):
        """
        测量轨枕组件的几何参数
        
        参数:
            cgt: 承轨台点云
            dyk: 单元块点云  
            test_mode: 是否保存中间结果
            is_right_side: 是否为右侧组件
            
        返回:
            dict: 包含s1, h, we测量值和置信度
        """
        logger.info(f"开始测量轨枕组件，{'右侧' if is_right_side else '左侧'}")
        
        # 数据预处理
        if len(cgt.points) == 0 or len(dyk.points) == 0:
            logger.error("输入点云为空")
            return None
        
        # 创建点云副本以避免修改原始数据
        cgt_copy = self._copy_point_cloud(cgt)
        dyk_copy = self._copy_point_cloud(dyk)
        
        # 并行执行三个测量
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_s1 = executor.submit(self.calculate_symmetry, cgt_copy, dyk_copy, test_mode, is_right_side)
            future_h = executor.submit(self.calculate_height, cgt_copy, dyk_copy, test_mode)
            future_we = executor.submit(self.calculate_side_width, cgt_copy, dyk_copy, test_mode, is_right_side)
            
            s1_result = future_s1.result()
            h_result = future_h.result()
            we_result = future_we.result()
    
        
        # 获取测量值（如果失败则使用默认值）
        s1 = s1_result['value'] if s1_result['success'] else 0.0
        h = h_result['value'] if h_result['success'] else 0.0
        we = we_result['value'] if we_result['success'] else 0.0
        
        logger.info(f"轨枕组件测量完成，s1: {s1:.2f}mm, h: {h:.2f}mm, we: {we:.2f}mm")
        
        return {
            'symmetry_s1': {'value': s1, 'unit': 'mm', 'tolerance': '±15mm'},
            'height_h': {'value': h, 'unit': 'mm'},
            'side_width_we': {'value': we, 'unit': 'mm'},
        }

# 独立测试代码
if __name__ == "__main__":
    # 测试模式：加载点云并进行测量
    import sys
    
    logger.info("开始独立测试测量模块")
    
    # 创建测量实例
    measurer = RailMeasurement()
    
    # 尝试加载测试点云
    try:
        # 尝试从当前目录加载
        cgt_path = "result/2512130211/cgt_r.ply"
        dyk_path = "result/2512130211/dyk_r.ply"
        
        # 如果不存在，尝试从分割结果_1目录加载
        if not os.path.exists(cgt_path):
            cgt_path = os.path.join("measurements_results\segmentation_results", cgt_path)
        if not os.path.exists(dyk_path):
            dyk_path = os.path.join("measurements_results\segmentation_results", dyk_path)
        
        # 加载点云
        cgt = o3d.io.read_point_cloud(cgt_path)
        dyk = o3d.io.read_point_cloud(dyk_path)
        
        logger.info(f"成功加载点云: cgt - {len(cgt.points)}点, dyk - {len(dyk.points)}点")
        
        # 执行测量（测试模式）
        results = measurer.measure_rail_components(cgt, dyk, test_mode=True,is_right_side=True)
        
        # 打印结果
        if results:
            print("\n测量结果:")
            print(f"对称度s1: {results['symmetry_s1']['value']:.2f} {results['symmetry_s1']['unit']}")
            print(f"承轨台高度h: {results['height_h']['value']:.2f} {results['height_h']['unit']}")
            print(f"侧面宽度we: {results['side_width_we']['value']:.2f} {results['side_width_we']['unit']}")
        
    except FileNotFoundError as e:
        logger.error(f"找不到测试点云文件: {e}")
        print("错误: 找不到测试点云文件，请确保cgt_l.ply和dyk_l.ply在正确位置")
    except Exception as e:
        logger.error(f"测试过程中出错: {e}")
        print(f"错误: 测试过程中发生错误: {str(e)}")
    
    logger.info("测量模块测试完成")