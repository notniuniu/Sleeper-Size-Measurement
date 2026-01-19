from tkinter import Y
import numpy as np
import open3d as o3d
import logging
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
from scipy import stats
from collections import Counter

def keep_mode_interval(data, method='kde', interval_width=0.2, visualize=False):
    """
    保留数据中出现频率最高的值所在的区间，而不是单一众数
    
    参数:
    data: 输入的数据列表或NumPy数组
    method: 使用的方法 ('kde'=核密度估计, 'freq'=频率区间)
    interval_width: 区间宽度（仅对freq方法有效）
    visualize: 是否显示可视化（需要matplotlib）
    
    返回:
    result: 保留在众数区间内的数据点
    mode_info: 关于众数区间的信息
    """
    data = np.array(data)
    
    if method == 'kde':
        # 方法1: 使用核密度估计找到密度最高的区域
        if len(data) > 1:
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(np.min(data), np.max(data), 1000)
            density = kde(x_range)
            
            # 找到密度最大值点
            max_density_idx = np.argmax(density)
            mode_center = x_range[max_density_idx]
            
            # 确定密度较高的区间（密度 > 最大密度的80%）
            threshold = 0.8 * np.max(density)
            high_density_indices = np.where(density >= threshold)[0]
            
            # 找到连续的高密度区间
            intervals = []
            start_idx = high_density_indices[0]
            for i in range(1, len(high_density_indices)):
                if high_density_indices[i] != high_density_indices[i-1] + 1:
                    intervals.append((x_range[start_idx], x_range[high_density_indices[i-1]]))
                    start_idx = high_density_indices[i]
            intervals.append((x_range[start_idx], x_range[high_density_indices[-1]]))
            
            # 选择最宽的区间作为主要众数区间
            main_interval = max(intervals, key=lambda x: x[1]-x[0])
            lower_bound, upper_bound = main_interval
            
        else:
            # 数据量太少，直接返回所有数据
            return data, {'method': 'kde', 'interval': (np.min(data), np.max(data)), 'note': 'insufficient data'}
    
    elif method == 'freq':
        # 方法2: 基于频率的区间分析
        if len(data) > 0:
            # 将数据分组到区间
            data_min, data_max = np.min(data), np.max(data)
            range_width = (data_max - data_min) * interval_width
            
            # 尝试不同的区间起点，找到包含最多数据点的区间
            best_interval = None
            max_count = 0
            
            for start in np.linspace(data_min, data_max - range_width, 50):
                end = start + range_width
                count = np.sum((data >= start) & (data <= end))
                
                if count > max_count or (count == max_count and best_interval is None):
                    max_count = count
                    best_interval = (start, end)
            
            lower_bound, upper_bound = best_interval
        else:
            return np.array([]), {'method': 'freq', 'interval': None, 'note': 'empty data'}
    
    # 过滤数据
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    
    # 准备返回信息
    mode_info = {
        'method': method,
        'interval': (float(lower_bound), float(upper_bound)),
        'interval_width': float(upper_bound - lower_bound),
        'original_count': len(data),
        'filtered_count': len(filtered_data),
        'retained_ratio': len(filtered_data) / len(data) if len(data) > 0 else 0
    }
    
    # 可视化（可选）
    if visualize and len(data) > 0:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.hist(data, bins=30, alpha=0.7, label='原始数据')
            plt.axvline(lower_bound, color='red', linestyle='--', label='众数区间边界')
            plt.axvline(upper_bound, color='red', linestyle='--')
            plt.xlabel('数值')
            plt.ylabel('频数')
            plt.title('原始数据分布')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.hist(filtered_data, bins=20, alpha=0.7, color='green', label='过滤后数据')
            plt.xlabel('数值')
            plt.ylabel('频数')
            plt.title('保留的众数区间数据')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("可视化需要matplotlib库，请先安装: pip install matplotlib")
    
    return filtered_data, mode_info

def extract_boundary_points_by_dynamic_bins(xy_points, num_bins=20, distance_threshold=10):
    """
    使用动态箱法提取点云的四条边界点
    
    参数:
    xy_points: numpy数组, 形状为(n, 2), 表示n个点的XY坐标
    num_bins: 整数, 分箱数量, 默认20
    distance_threshold: 浮点数, 距离边界的阈值
    
    返回:
    x_min_points: X最小值边界点列表
    x_max_points: X最大值边界点列表  
    y_min_points: Y最小值边界点列表
    y_max_points: Y最大值边界点列表
    """
    # 输入验证
    if not isinstance(xy_points, np.ndarray) or xy_points.shape[1] != 2:
        raise ValueError("xy_points必须是形状为(n, 2)的numpy数组")
    
    # 分离X和Y坐标
    x_coords = xy_points[:, 0]
    y_coords = xy_points[:, 1]
    
    # 初始化结果列表
    x_min_points = []
    x_max_points = []
    y_min_points = []
    y_max_points = []
    
    # 提取Y方向边界点（沿X轴分箱）
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    x_bins = np.linspace(x_min, x_max, num_bins + 1)  # 生成X轴箱边界
        # 提取X方向边界点（沿Y轴分箱）
    y_min_val, y_max_val = np.min(y_coords), np.max(y_coords)
    y_bins = np.linspace(y_min_val, y_max_val, num_bins + 1)  # 生成Y轴箱边界
    
    for i in range(num_bins):
        # 找出当前X箱内的点
        mask = (x_coords >= x_bins[i]) & (x_coords <= x_bins[i + 1])
        points_in_bin = xy_points[mask]
        
        if len(points_in_bin) > 0:
            # 提取Y值在边界距离阈值内的点
            y_values = points_in_bin[:, 1]
            y_min_in_bin = np.min(y_values)
            y_max_in_bin = np.max(y_values)
            
            # 计算Y方向的差距，如果差距很小，则只输出一个边缘
            y_range = y_max_in_bin - y_min_in_bin
            
            if y_range <= distance_threshold * 2:  # 当差距小于等于2倍阈值时，认为是靠近一个边
                # 计算y_min_in_bin到全局y_min的距离和y_max_in_bin到全局y_max的距离
                distance_to_min = abs(y_min_in_bin - y_min_val)
                distance_to_max = abs(y_max_in_bin - y_max_val)
                
                # 选择距离更近的那个边缘作为代表
                if distance_to_min <= distance_to_max:
                    # 更靠近全局Y最小值，添加到Y最小边缘点
                    min_y_mask = (y_values - y_min_in_bin) <= distance_threshold
                    min_y_points = points_in_bin[min_y_mask]
                    y_min_points.extend(min_y_points)
                    logger.debug(f"Y箱内点云范围很小({y_range:.4f})，更靠近Y最小值边缘，只添加Y最小边缘点")
                else:
                    # 更靠近全局Y最大值，添加到Y最大边缘点
                    max_y_mask = (y_max_in_bin - y_values) <= distance_threshold
                    max_y_points = points_in_bin[max_y_mask]
                    y_max_points.extend(max_y_points)
                    logger.debug(f"Y箱内点云范围很小({y_range:.4f})，更靠近Y最大值边缘，只添加Y最大边缘点")
            else:
                # 正常情况下分别提取Y最大和最小边缘点
                # 提取距离Y最小值在阈值内的点
                min_y_mask = (y_values - y_min_in_bin) <= distance_threshold
                min_y_points = points_in_bin[min_y_mask]
                
                # 提取距离Y最大值在阈值内的点
                max_y_mask = (y_max_in_bin - y_values) <= distance_threshold
                max_y_points = points_in_bin[max_y_mask]
                
                # 添加到结果列表
                y_min_points.extend(min_y_points)
                y_max_points.extend(max_y_points)
    

    
    for i in range(num_bins):
        # 找出当前Y箱内的点
        mask = (y_coords >= y_bins[i]) & (y_coords <= y_bins[i + 1])
        points_in_bin = xy_points[mask]
        
        if len(points_in_bin) > 0:
            # 提取X值在边界距离阈值内的点
            x_values = points_in_bin[:, 0]
            x_min_in_bin = np.min(x_values)
            x_max_in_bin = np.max(x_values)
            
            # 计算X方向的差距，如果差距很小，则只输出一个边缘
            x_range = x_max_in_bin - x_min_in_bin
            
            if x_range <= distance_threshold * 2:  # 当差距小于等于2倍阈值时，认为是靠近一个边
                # 计算x_min_in_bin到全局x_min的距离和x_max_in_bin到全局x_max的距离
                distance_to_min = abs(x_min_in_bin - x_min)
                distance_to_max = abs(x_max_in_bin - x_max)
                
                # 选择距离更近的那个边缘作为代表
                if distance_to_min <= distance_to_max:
                    # 更靠近全局X最小值，添加到X最小边缘点
                    min_x_mask = (x_values - x_min_in_bin) <= distance_threshold
                    min_x_points = points_in_bin[min_x_mask]
                    x_min_points.extend(min_x_points)
                    logger.debug(f"X箱内点云范围很小({x_range:.4f})，更靠近X最小值边缘，只添加X最小边缘点")
                else:
                    # 更靠近全局X最大值，添加到X最大边缘点
                    max_x_mask = (x_max_in_bin - x_values) <= distance_threshold
                    max_x_points = points_in_bin[max_x_mask]
                    x_max_points.extend(max_x_points)
                    logger.debug(f"X箱内点云范围很小({x_range:.4f})，更靠近X最大值边缘，只添加X最大边缘点")
            else:
                # 正常情况下分别提取X最大和最小边缘点
                # 提取距离X最小值在阈值内的点
                min_x_mask = (x_values - x_min_in_bin) <= distance_threshold
                min_x_points = points_in_bin[min_x_mask]
                
                # 提取距离X最大值在阈值内的点
                max_x_mask = (x_max_in_bin - x_values) <= distance_threshold
                max_x_points = points_in_bin[max_x_mask]
                
                # 添加到结果列表
                x_min_points.extend(min_x_points)
                x_max_points.extend(max_x_points)
    
    # 转换为numpy数组
    x_min_points = np.array(x_min_points)
    x_max_points = np.array(x_max_points)
    y_min_points = np.array(y_min_points)
    y_max_points = np.array(y_max_points)
    
    # 记录日志
    logger.info(f"动态箱法提取边界点结果: X最小点数量={len(x_min_points)}, X最大点数量={len(x_max_points)}, "
               f"Y最小点数量={len(y_min_points)}, Y最大点数量={len(y_max_points)}")
    
    return x_min_points, x_max_points, y_min_points, y_max_points

def calculate_parallelogram_length_width(pcd, visualize=False, output_file="单元块边缘线.ply",
                                         y_max_distance=float('inf'),y_min_distance=0,
                                         x_max_distance=float('inf'),x_min_distance=0):
    """
    在XY平面上分析并计算点云的长度和宽度
    
    Args:
        pcd: Open3D格式的点云对象
        visualize: 是否可视化结果，默认为False
        output_file: 边缘线输出文件路径，默认为None
        
        y_max_distance: Y轴最大距离阈值，默认为无穷大
        y_min_distance: Y轴最小距离阈值，默认为0
        x_max_distance: X轴最大距离阈值，默认为无穷大
        x_min_distance: X轴最小距离阈值，默认为0
        
    Returns:
        tuple: (长度, 宽度)，保留两位小数
    """
    try:
        # 检查点云是否有效
        if pcd is None or len(pcd.points) == 0:
            raise ValueError("输入点云为空")
        
        if len(pcd.points) < 4:
            raise ValueError(f"点云点数不足，需要至少4个点，当前仅有{len(pcd.points)}个点")
        
        # 1. 提取XY坐标信息
        points = np.asarray(pcd.points)
        xy_points = points[:, :2]  # 仅保留XY坐标
        logger.info(f"提取到{xy_points.shape[0]}个点的XY坐标")
        
        # 初始提取边缘点 - 使用动态箱法提取每个箱内距离边界2mm内的点
        x_min_points, x_max_points, y_min_points, y_max_points = extract_boundary_points_by_dynamic_bins(xy_points, distance_threshold=1)
        
        x_min, x_max, y_min, y_max = np.min(xy_points[:, 0]), np.max(xy_points[:, 0]), np.min(xy_points[:, 1]), np.max(xy_points[:, 1])
        # 标记是否需要扩展边缘宽度  
        need_expand = True

        # 4. 使用最小二乘法拟合四条边缘直线
        def fit_edge_line(edge_points, is_x_direction=True):
            """
            拟合边缘直线，并确保近似与坐标轴平行
            
            Args:
                edge_points: 边缘点集
                is_x_direction: 是否为X方向边缘（垂直于X轴）
                
            Returns:
                tuple: (a, b, c) 直线参数 ax + by + c = 0
            """
            if len(edge_points) < 2:
                # 如果点太少，返回默认直线
                if is_x_direction:
                    return 1, 0, -np.median(edge_points[:, 0]) if len(edge_points) > 0 else 0
                else:
                    return 0, 1, -np.median(edge_points[:, 1]) if len(edge_points) > 0 else 0
                
            # 使用最小二乘法拟合直线
            X = edge_points[:, 0].reshape(-1, 1)
            y = edge_points[:, 1]
            
            model = LinearRegression()
            model.fit(X, y)
            
            # 获取斜率和截距
            slope = model.coef_[0]
            intercept = model.intercept_
            
            # 判断是否需要强制与坐标轴平行
            slope_threshold = 200
            if abs(slope) > slope_threshold:
                logger.warning(f"拟合直线斜率为{slope:.4f}，超过阈值{slope_threshold}，强制调整")
                if is_x_direction:
                    # X方向边缘应垂直于X轴，即x=constant
                    x_mean = np.median(edge_points[:, 0])
                    return 1, 0, -x_mean
                else:
                    # Y方向边缘应垂直于Y轴，即y=constant
                    y_mean = np.median(edge_points[:, 1])
                    return 0, 1, -y_mean
            
            # 转换为ax + by + c = 0形式
            a = -slope
            b = 1
            c = -intercept
            
            # 标准化
            norm = np.sqrt(a**2 + b**2)
            a, b, c = a/norm, b/norm, c/norm
            
            return a, b, c
        

        
        
        # 计算点集之间的距离
        def calculate_point_set_distances(points1, points2, is_x_direction=True, y_tolerance_ratio=0.05,max_distance=float('inf'),min_distance=0):
            """
            计算点集之间满足Y轴坐标范围条件的点对距离
            
            Args:
                points1, points2: 点集
                is_x_direction: 是否为X轴方向
                y_tolerance_ratio: Y轴坐标容差比例
                max_distance: 最大距离阈值
                min_distance: 最小距离阈值
                
            Returns:
                dict: 包含处理后的平均、最大、最小、众数距离的字典
            """
            # 声明使用外部变量
            nonlocal need_expand, x_min, x_max, y_min, y_max

            if is_x_direction:
                # 计算Y轴坐标容差
                y_range = max(np.max(points1[:, 1]), np.max(points2[:, 1])) - min(np.min(points1[:, 1]), np.min(points2[:, 1]))
                y_tolerance = y_range * y_tolerance_ratio
                # 寻找匹配的点对
                matched_distances = []
                
                for p1 in points1:
                    # 找到Y坐标在容差范围内的点
                    mask = np.abs(points2[:, 1] - p1[1]) <= y_tolerance
                    matching_points = points2[mask]
                    
                    if len(matching_points) > 0:
                        distances = abs(matching_points[:, 0] - p1[0])
                        # 选择最小距离作为匹配距离
                        if min_distance < np.min(distances) and np.min(distances) < max_distance:
                            matched_distances.append(np.min(distances))

                if len(matched_distances) > 0:
                    # 计算统计指标
                    mean_distance = np.mean(matched_distances)
                    max_distance = np.max(matched_distances)
                    min_distance = np.min(matched_distances)
                    # mode_distance = stats.mode(matched_distances, keepdims=True).mode[0]
                    filtered_count = len(matched_distances)
                else:
                    logger.warning(f"未找到Y坐标在容差范围内的匹配点: p1={p1}")
                    need_expand = True  # 标记需要扩展边缘宽度

            else:
                # 计算X轴坐标容差
                x_range = max(np.max(points1[:, 0]), np.max(points2[:, 0])) - min(np.min(points1[:, 0]), np.min(points2[:, 0]))
                x_tolerance = x_range * y_tolerance_ratio
                                # 寻找匹配的点对
                matched_distances = []
                
                for p1 in points1:
                    # 找到X坐标在容差范围内的点
                    mask = np.abs(points2[:, 0] - p1[0]) <= x_tolerance
                    matching_points = points2[mask]
                    
                    if len(matching_points) > 0:
                        # 计算X方向距离
                        distances = abs(matching_points[:, 1] - p1[1])
                        if min_distance < np.min(distances) and np.min(distances) < max_distance:
                            matched_distances.append(np.min(distances))
                if len(matched_distances) > 0:
                    # 计算统计指标
                    mean_distance = np.mean(matched_distances)
                    max_distance = np.max(matched_distances)
                    min_distance = np.min(matched_distances)
                    # mode_distance = stats.mode(matched_distances, keepdims=True).mode[0]
                    filtered_count = len(matched_distances)
                else:
                    logger.warning(f"未找到Y坐标在容差范围内的匹配点: p1={p1}")
                    need_expand = True  # 标记需要扩展边缘宽度
            
            
            
            if not matched_distances:
                return {
                    'mean_distance': 0,
                    'max_distance': 0,
                    'min_distance': 0,
                    'mode_distance': 0,
                    'filtered_count': 0
                }
            
            matched_distances = np.array(matched_distances)
            
            # 离群值剔除（四分位数方法）
            # Q1 = np.percentile(matched_distances, 25)
            # Q3 = np.percentile(matched_distances, 75)
            # IQR = Q3 - Q1
            # lower_bound = Q1 - 1.5 * IQR
            # upper_bound = Q3 + 1.5 * IQR
            
            # filtered_distances = matched_distances[(matched_distances >= lower_bound) & (matched_distances <= upper_bound)]

            filtered_freq, info_freq = keep_mode_interval(matched_distances, method='freq', interval_width=0.3, visualize=False)
            # print(f"众数区间: {info_freq['interval']}")
            # print(f"保留数据: {info_freq['filtered_count']} 个点 ({info_freq['retained_ratio']*100:.1f}%)")

            # filtered_kde, info_kde = keep_mode_interval(matched_distances, method='kde', visualize=True)
            # print(f"众数区间: {info_kde['interval']}")
            # print(f"保留数据: {info_kde['filtered_count']} 个点 ({info_kde['retained_ratio']*100:.1f}%)")
            
            # 计算众数
            if len(filtered_freq) > 0:
                hist, bins = np.histogram(filtered_freq, bins='auto')
                max_bin_index = np.argmax(hist)
                mode_distance = (bins[max_bin_index] + bins[max_bin_index + 1]) / 2
            else:
                mode_distance = 0
            
            return {
                'mean_distance': np.mean(filtered_freq) if len(filtered_freq) > 0 else 0,
                'max_distance': np.max(filtered_freq) if len(filtered_freq) > 0 else 0,
                'min_distance': np.min(filtered_freq) if len(filtered_freq) > 0 else 0,
                'mode_distance': mode_distance,
                'filtered_count': len(filtered_freq)
            }
        
        # 主循环：处理边缘宽度扩展
        max_attempts = 100  # 最大尝试次数
        current_attempt = 0
        
        while current_attempt < max_attempts:
            # 拟合四条边缘直线
            x_min_line = fit_edge_line(x_min_points, is_x_direction=True)
            x_max_line = fit_edge_line(x_max_points, is_x_direction=True)
            y_min_line = fit_edge_line(y_min_points, is_x_direction=False)
            y_max_line = fit_edge_line(y_max_points, is_x_direction=False)
            
            logger.info(f"尝试{current_attempt + 1}/{max_attempts}")
            logger.info(f"X最小边缘直线: {x_min_line}")
            logger.info(f"X最大边缘直线: {x_max_line}")
            logger.info(f"Y最小边缘直线: {y_min_line}")
            logger.info(f"Y最大边缘直线: {y_max_line}")
            
            # 计算X轴方向的距离（宽度）
            x_point_distances = calculate_point_set_distances(x_min_points, x_max_points, is_x_direction=True,max_distance=x_max_distance,min_distance=x_min_distance)
            
            # 检查是否需要扩展边缘宽度
            if need_expand:
                current_attempt += 1
                need_expand = False  # 重置标志
                # 重新提取边缘点
                x_min_points, x_max_points, y_min_points, y_max_points = extract_boundary_points_by_dynamic_bins(xy_points, distance_threshold=current_attempt)
            else:
                # 如果不需要扩展，跳出循环
                break
        
        # 记录最终的距离计算结果
        logger.info(f"X轴方向点集距离:")
        logger.info(f"  平均距离: {x_point_distances['mean_distance']:.4f}")
        logger.info(f"  最大距离: {x_point_distances['max_distance']:.4f}")
        logger.info(f"  最小距离: {x_point_distances['min_distance']:.4f}")
        logger.info(f"  众数距离: {x_point_distances['mode_distance']:.4f}")
        logger.info(f"  有效点对数量: {x_point_distances['filtered_count']}")
        
        # 计算Y轴方向的距离（长度）
        y_point_distances = calculate_point_set_distances(y_min_points, y_max_points, is_x_direction=False,
                                                          max_distance=y_max_distance,min_distance=y_min_distance)
        
        logger.info(f"Y轴方向点集距离:")
        logger.info(f"  平均距离: {y_point_distances['mean_distance']:.4f}")
        logger.info(f"  最大距离: {y_point_distances['max_distance']:.4f}")
        logger.info(f"  最小距离: {y_point_distances['min_distance']:.4f}")
        logger.info(f"  众数距离: {y_point_distances['mode_distance']:.4f}")
        logger.info(f"  有效点对数量: {y_point_distances['filtered_count']}")
        
        # 使用点集平均距离作为最终的宽度和长度
        width = x_point_distances['mean_distance']
        length = y_point_distances['mean_distance']
        
        # # 确保长度大于宽度
        if length < width:
            length, width = width, length
        
        # 6. 保留两位小数
        length = round(length, 2)
        width = round(width, 2)
        
        logger.info(f"计算结果 - 长度: {length}, 宽度: {width}")

        
        # 生成边缘线点并保存为ply文件（如果需要）
        if output_file:
            # 计算点云的平均Z值
            z_mean = np.mean(points[:, 2])
            
            # 生成边缘线上的点
            def generate_line_points(line_params, min_val, max_val, num_points=50):
                """
                根据直线参数生成直线上的点
                """
                a, b, c = line_params
                line_points = []
                
                if abs(a) > abs(b):  # 垂直直线，x为常数
                    x_val = -c / a
                    y_vals = np.linspace(min_val, max_val, num_points)
                    for y_val in y_vals:
                        line_points.append([x_val, y_val])
                else:  # 水平直线或倾斜直线，y = (-a/b)x - c/b
                    x_vals = np.linspace(min_val, max_val, num_points)
                    for x_val in x_vals:
                        y_val = (-a * x_val - c) / b
                        line_points.append([x_val, y_val])
                
                return np.array(line_points)
            
            # 获取旋转后的点云范围
            y_range = y_max - y_min
            x_range = x_max - x_min
            
            # 生成四条边缘线上的点
            x_min_line_points = generate_line_points(x_min_line, y_min, y_max)
            x_max_line_points = generate_line_points(x_max_line, y_min, y_max)
            y_min_line_points = generate_line_points(y_min_line, x_min, x_max)
            y_max_line_points = generate_line_points(y_max_line, x_min, x_max)
            
            # 合并所有边缘线点
            original_xy_points = np.vstack([x_min_points, x_max_points
            ,y_min_points,y_max_points])
            
            # 将点旋转回原始坐标系
            # inv_rotation_matrix = np.linalg.inv(rotation_matrix)
            # original_xy_points = np.dot(all_line_points, inv_rotation_matrix) + pca.mean_
            
            # 添加Z坐标（使用平均Z值）
            original_3d_points = np.column_stack([original_xy_points, 
                                                np.full(len(original_xy_points), z_mean)])
            
            # 创建边缘线点云
            edge_line_pcd = o3d.geometry.PointCloud()
            edge_line_pcd.points = o3d.utility.Vector3dVector(original_3d_points)
            
            # 为边缘线点云添加颜色（红色）
            edge_colors = np.zeros((len(original_3d_points), 3))
            edge_colors[:, 0] = 1.0  # 红色
            edge_line_pcd.colors = o3d.utility.Vector3dVector(edge_colors)
            
            # 保存为ply文件
            o3d.io.write_point_cloud(output_file, edge_line_pcd)
            logger.info(f"边缘线已保存至: {output_file}")
        
        # 可视化（如果需要）
        if visualize:
            # 创建可视化点云
            visualize_pcd = o3d.geometry.PointCloud()
            visualize_pcd.points = o3d.utility.Vector3dVector(points)
            
            # 为点云添加颜色
            colors = np.zeros((len(points), 3))
            colors[:, 0] = 0.5  # 灰色点云
            visualize_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 如果有边缘线文件，加载并显示
            geometries = [visualize_pcd]
            if output_file and os.path.exists(output_file):
                edge_pcd = o3d.io.read_point_cloud(output_file)
                geometries.append(edge_pcd)
            
            # 显示结果
            o3d.visualization.draw_geometries(geometries, 
                                              window_name=f"平行四边形测量 - 长度: {length}, 宽度: {width}")
        
        
        # 检查长宽值是否为null，如果是则使用包围盒尺寸替代
        if length == 0 or width == 0:
            # 计算edge_line_pcd的包围盒
            aabb = edge_line_pcd.get_axis_aligned_bounding_box()
            # 获取包围盒的尺寸（x, y, z）
            bbox_size = aabb.get_extent()
            
            # 使用包围盒的x和y尺寸替代null值
            if length == 0:
                length = round(bbox_size[0], 2)
                logger.info(f"长度计算结果为0，使用包围盒长度: {length}")
            if width == 0:
                width = round(bbox_size[1], 2)
                logger.info(f"宽度计算结果为0，使用包围盒宽度: {width}")
        
        return length, width
        
    except ValueError as e:
        logger.error(f"值错误: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"计算过程中发生错误: {str(e)}")
        raise

# 测试函数
def test_calculate_parallelogram_length_width():
    """
    测试calculate_parallelogram_length_width函数
    """
    try:
        # 创建一个简单的平行四边形点云进行测试
        # 定义四个顶点
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1.5, 1, 0],
            [0.5, 1, 0]
        ])
        
        # 创建点云
        pcd = o3d.io.read_point_cloud("test_results/侧面宽度_测量线_左侧.ply")

        # 计算长度和宽度
        length, width = calculate_parallelogram_length_width(pcd, visualize=True,output_file = '单元块_底部边缘线.ply',
        y_max_distance=float('inf'),y_min_distance=0,x_max_distance=float('inf'),x_min_distance=0)
        
        print(f"测试结果: 长度 = {length}, 宽度 = {width}")

        
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    # 运行测试
    test_calculate_parallelogram_length_width()