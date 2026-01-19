import open3d as o3d
import numpy as np
import os
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('measurement_2')

class RailInclinationMeasurement:
    """轨枕承轨台倾斜角测量类"""
    
    def __init__(self, output_dir="test_result"):
        """
        初始化测量类
        
        Args:
            output_dir (str): 中间结果输出目录
        """
        self.output_dir = output_dir
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")
        
        # 测量参数配置
        self.min_points_threshold = 20  # 最小点数阈值
        self.top_percentile = 70  # 筛选Z值最高的百分比
        self.normalization_factor = 1 # 毫米转换因子
    
    def _extract_top_points(self, pcd, top_percentile=70):
        """
        提取点云中Z值最高的指定百分比的点
        
        Args:
            pcd: 输入点云
            top_percentile: 提取Z值最高的百分比
            
        Returns:
            筛选后的点云
        """
        points = np.asarray(pcd.points)
        if len(points) < self.min_points_threshold:
            logger.warning(f"点云点数过少: {len(points)}点，无法提取上表面点")
            return None
        
        # 获取Z坐标
        z_coords = points[:, 2]
        # 计算Z值的百分位数阈值
        z_threshold = np.percentile(z_coords, 100 - top_percentile)
        # 筛选Z值高于阈值的点
        top_points_mask = z_coords >= z_threshold
        top_points = points[top_points_mask]
        
        if len(top_points) < self.min_points_threshold:
            logger.warning(f"上表面点数量不足: {len(top_points)}点")
            return None
        
        # 创建新的点云
        top_pcd = o3d.geometry.PointCloud()
        top_pcd.points = o3d.utility.Vector3dVector(top_points)
        logger.info(f"成功提取上表面点: {len(top_points)}点 (Z值最高{top_percentile}%)")
        
        return top_pcd
    
    def _fit_plane_ransac(self, pcd, distance_threshold=None, ransac_n=3, num_iterations=1000):
        """
        使用RANSAC算法拟合平面
        
        Args:
            pcd: 输入点云
            distance_threshold: 距离阈值，None表示自动计算
            ransac_n: RANSAC采样点数
            num_iterations: 迭代次数
            
        Returns:
            (plane_model, inliers): 平面模型和平面内点索引
        """
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
    
    def _filter_outliers(self, pcd, distance_threshold=0.005):
        """
        去除点云中的离群点，保留平面内的点
        
        Args:
            pcd: 输入点云
            distance_threshold: 距离阈值
            
        Returns:
            过滤后的点云
        """
        # 先拟合平面获取初始模型
        plane_model, inliers = self._fit_plane_ransac(pcd)
        if plane_model is None:
            return pcd
        
        # 获取平面内的点
        filtered_pcd = pcd.select_by_index(inliers)
        logger.info(f"离群点过滤完成: 从 {len(pcd.points)} 点到 {len(filtered_pcd.points)} 点")
        
        return filtered_pcd
    
    def _save_point_cloud(self, pcd, filename, color=None):
        """
        保存点云到文件
        
        Args:
            pcd: 点云对象
            filename: 文件名
            color: 点云颜色 [r, g, b]
        """
        if pcd is None or len(pcd.points) == 0:
            logger.warning(f"无法保存空点云: {filename}")
            return False
        
        try:
            # 复制点云以避免修改原始点云
            save_pcd = o3d.geometry.PointCloud()
            save_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
            
            # 设置颜色
            if color is not None:
                colors = np.tile(np.array(color), (len(pcd.points), 1))
                save_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 保存文件
            file_path = os.path.join(self.output_dir, filename)
            o3d.io.write_point_cloud(file_path, save_pcd)
            logger.info(f"点云保存成功: {file_path}")
            return True
        except Exception as e:
            logger.error(f"点云保存失败 {filename}: {e}")
            return False
    
    def _visualize_normal_vector(self, normal_vector, plane_center, filename):
        """
        可视化平面法向量
        
        Args:
            normal_vector: 平面法向量
            plane_center: 平面中心点
            filename: 文件名
        """
        # 计算法向量的可视化点
        # 缩放法向量以便于观察
        scale = 50  # 缩放因子
        scaled_normal = normal_vector * scale
        
        # 起点（平面中心）
        start_point = plane_center
        # 终点（平面中心 + 缩放后的法向量）
        end_point = plane_center + scaled_normal
        
        # 创建法向量的点云（使用多个点来表示线段）
        num_points = 50
        normal_points = np.array([start_point + t * scaled_normal for t in np.linspace(0, 1, num_points)])
        
        # 创建点云并设置颜色
        normal_pcd = o3d.geometry.PointCloud()
        normal_pcd.points = o3d.utility.Vector3dVector(normal_points)
        normal_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.array([1, 0, 0]), (num_points, 1)))
        
        # 保存点云
        self._save_point_cloud(normal_pcd, filename)
        
        logger.info(f"法向量可视化成功，长度缩放因子: {scale}")
    
    def calculate_inclination_angle(self, cgt, test_mode=False):
        """
        计算承轨台倾斜角
        
        Args:
            cgt: 承轨台点云数据
            test_mode: 是否保存中间结果
            
        Returns:
            dict: 包含倾斜角值和相关信息
        """
        start_time = time.time()
        logger.info("开始计算承轨台倾斜角")
        
        result = {
            'value': 0.0,
            'success': False,
            'plane_model': None,
            'normal_vector': None
        }
        
        # 步骤1: 提取承轨台上表面点云
        top_points = self._extract_top_points(cgt, top_percentile=self.top_percentile)
        if top_points is None:
            logger.error("提取上表面点失败")
            return result
        
        # 步骤2: 平面拟合获取上表面模型
        plane_model, inliers = self._fit_plane_ransac(top_points)
        if plane_model is None:
            logger.error("平面拟合失败")
            return result
        
        # 保存平面模型
        result['plane_model'] = plane_model
        
        # 获取平面内点
        plane_points = top_points.select_by_index(inliers)
        
        # 步骤3: 离群点去除精炼平面参数
        refined_plane_points = self._filter_outliers(plane_points)
        
        # 重新拟合平面以获取更准确的参数
        refined_plane_model, _ = self._fit_plane_ransac(refined_plane_points)
        if refined_plane_model is not None:
            plane_model = refined_plane_model
            result['plane_model'] = plane_model
        
        # 获取平面法向量
        normal_vector = np.array(plane_model[:3])
        # 确保法向量指向正方向（Z分量为正）
        if normal_vector[2] < 0:
            normal_vector = -normal_vector
        result['normal_vector'] = normal_vector
        
        # 计算平面中心点
        plane_center = np.mean(np.asarray(refined_plane_points.points), axis=0)
        
        # 步骤4: 计算倾斜角
        # 法向量在YZ平面上的投影
        normal_projection_yz = normal_vector[1:3]  # y和z分量
        # Y轴单位向量
        y_axis = np.array([0, 1])
        
        # 计算法向量投影与Y轴的夹角
        # 单位化向量
        normal_projection_yz_unit = normal_projection_yz / np.linalg.norm(normal_projection_yz)
        
        # 计算夹角（弧度）
        cos_theta = np.dot(normal_projection_yz_unit, y_axis)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保数值稳定性
        theta_rad = np.arccos(cos_theta)
        
        # 转换为角度
        theta_deg = np.rad2deg(theta_rad)
        
        # 倾斜角计算公式：90°减去夹角
        inclination_angle =  theta_deg
        result['value'] = inclination_angle
        result['success'] = True
        
        # 步骤5: 保存中间结果（测试模式）
        if test_mode:
            # 保存精炼后的平面内点
            self._save_point_cloud(refined_plane_points, "倾斜角_承轨台平面.ply", [0, 1, 0])
            
            # 可视化法向量
            self._visualize_normal_vector(normal_vector, plane_center, "倾斜角_承轨台法向量.ply")
        
        logger.info(f"承轨台倾斜角计算完成: {inclination_angle:.2f}°, 耗时: {time.time()-start_time:.2f}s")
        return result

# 示例使用代码
if __name__ == "__main__":
    # 创建测量对象
    measurer = RailInclinationMeasurement()
    
    # 示例：加载点云并计算倾斜角
    try:
        # 这里需要替换为实际的点云文件路径
        pcd = o3d.io.read_point_cloud("measurements_results\segmentation_results\cgt_l.ply")
        result = measurer.calculate_inclination_angle(pcd, test_mode=True)
        if result['success']:
            print(f"倾斜角: {result['value']:.2f}°")
        else:
            print("计算失败")
        pass
    except Exception as e:
        logger.error(f"示例运行失败: {e}")