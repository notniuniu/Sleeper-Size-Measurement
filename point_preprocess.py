import struct
from tkinter import N
import numpy as np
import os
from tqdm import tqdm
import open3d as o3d
import cv2

def load_and_process_points(input_file, batch_size=100000):
    """
    优化版：从二进制文件读取点云数据并进行处理（批量读取+向量化解析）

    Args:
        input_file: 输入二进制文件路径
        batch_size: 批量读取的记录数（越大越快，注意内存限制）

    Returns:
        numpy.ndarray: 处理后的点云数据，形状为(n_points, 3)
    """
    try:
        # 定义记录格式和大小
        record_format = '3fi'  # 3个float(4*3=12字节), 1个int(4字节) → 总计16字节
        record_size = struct.calcsize(record_format)
        float_size = struct.calcsize('f')
        int_size = struct.calcsize('i')

        # 获取文件基本信息
        file_size = os.path.getsize(input_file)
        total_records = file_size // record_size
        if file_size % record_size != 0:
            print(f"警告：文件大小({file_size}字节)不是记录大小({record_size}字节)的整数倍，尾部数据将被忽略")

        print(f"文件信息：大小={file_size / 1024 / 1024:.2f}MB, 预计记录数={total_records}, 记录大小={record_size}字节")

        if total_records == 0:
            print("错误：文件无有效记录")
            return np.array([])

        # 预计算批量读取的参数
        bytes_per_batch = batch_size * record_size
        total_batches = (total_records + batch_size - 1) // batch_size  # 向上取整

        # 预分配内存（可选，进一步提升速度）
        all_points = []
        error_count = 0

        # 批量读取+解析
        with open(input_file, 'rb') as file:
            with tqdm(total=total_records, desc="读取点云数据", unit="记录") as pbar:
                for batch_idx in range(total_batches):
                    # 计算当前批次的记录数
                    start_record = batch_idx * batch_size
                    end_record = min((batch_idx + 1) * batch_size, total_records)
                    current_batch_size = end_record - start_record
                    if current_batch_size <= 0:
                        break

                    # 批量读取字节数据
                    batch_bytes = file.read(current_batch_size * record_size)
                    if len(batch_bytes) != current_batch_size * record_size:
                        print(f"警告：批次{batch_idx + 1}读取字节数不匹配，提前终止读取")
                        break

                    # --------------------------
                    # 核心优化：向量化解析（替代循环unpack）
                    # --------------------------
                    # 将字节数据转换为numpy数组（按原始字节顺序）
                    # float: 4字节，int:4字节 → 每个记录结构：fff i
                    dtype = np.dtype([
                        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('extra', 'i4')
                    ])
                    # 调整字节顺序（根据实际文件的端序，默认小端）
                    dtype = dtype.newbyteorder('<')

                    # 解析整个批次为结构化数组
                    try:
                        batch_data = np.frombuffer(batch_bytes, dtype=dtype)

                        # 提取xyz并过滤NaN的Z值
                        xyz = np.column_stack([batch_data['x'], batch_data['y'], batch_data['z']])
                        # 过滤Z为NaN的点（向量化操作，远快于循环）
                        valid_mask = ~np.isnan(xyz[:, 2])
                        valid_xyz = xyz[valid_mask]

                        all_points.append(valid_xyz)

                    except Exception as e:
                        error_count += current_batch_size
                        if error_count <= 10 * batch_size:
                            print(f"批次 {batch_idx + 1} 解析错误: {e}")
                        elif error_count == 11 * batch_size:
                            print("更多错误批次将不再显示...")

                    # 更新进度条
                    pbar.update(current_batch_size)

        # 合并所有批次的有效点
        if not all_points:
            print("未读取到有效的点云数据")
            return np.array([])

        points_array = np.vstack(all_points)
        valid_count = len(points_array)
        total_processed = total_records - error_count

        # 输出读取统计信息
        print(
            f"\n读取完成: 总记录数 {total_records} → 成功解析 {total_processed} 条 → 有效点（非NaN Z） {valid_count} 个 → 解析错误 {error_count} 条")

        # 打印坐标范围
        print_point_ranges(points_array, "原始数据")

        # 检查并过滤异常大的值
        has_large_values = np.any(np.abs(points_array) > 10000)
        if has_large_values:
            print("\n警告：检测到异常大的坐标值，可能是格式不匹配!")
            points_array = filter_large_values(points_array)
            if len(points_array) == 0:
                print("所有点都被过滤掉了")
                return np.array([])

        return points_array

    except Exception as e:
        print(f"处理过程中出错: {e}")
        return np.array([])


def print_point_ranges(points_array, prefix=""):
    """
    打印点云数据的坐标范围
    
    Args:
        points_array: 点云数据数组
        prefix: 输出前缀信息
    """
    if prefix:
        prefix = f"{prefix} - "
    
    print(f"\n{prefix}坐标范围：")
    print(f"X: min={points_array[:,0].min():.4f}, max={points_array[:,0].max():.4f}")
    print(f"Y: min={points_array[:,1].min():.4f}, max={points_array[:,1].max():.4f}")
    print(f"Z: min={points_array[:,2].min():.4f}, max={points_array[:,2].max():.4f}")


def filter_large_values(points_array, threshold=10000):
    """
    过滤异常大的坐标值
    
    Args:
        points_array: 原始点云数据数组
        threshold: 过滤阈值，坐标绝对值超过此值的点将被过滤
    
    Returns:
        numpy.ndarray: 过滤后的点云数据
    """
    original_count = len(points_array)
    # 创建掩码，保留所有坐标绝对值小于阈值的点
    valid_mask = np.all(np.abs(points_array) < threshold, axis=1)
    # 应用掩码过滤点
    filtered_array = points_array[valid_mask]
    filtered_count = len(filtered_array)
    
    print(f"已过滤异常值，保留 {filtered_count}/{original_count} 个有效点")
    print_point_ranges(filtered_array, "过滤后数据")
    
    return filtered_array


def transform_points(points, transformation_matrix):
    """
    对点云数据应用坐标变换矩阵
    
    Args:
        points: n×3的numpy数组，表示点云数据，每行对应一个点的x、y、z坐标
        transformation_matrix: 4×4的numpy数组，表示变换矩阵
    
    Returns:
        numpy.ndarray: 变换后的点云数据，形状为(n×3)
    """
    # 检查输入参数
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数据必须是n×3的数组")
    
    if transformation_matrix.shape != (4, 4):
        raise ValueError("变换矩阵必须是4×4的数组")
    
    # 将点云转换为齐次坐标 (n×4)，添加1作为第四个分量
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # 使用矩阵乘法进行坐标变换（利用numpy的向量化操作提高效率）
    # 需要转置以匹配矩阵乘法的要求，然后再转置回来
    transformed_points_homogeneous = np.dot(transformation_matrix, points_homogeneous.T).T
    
    # 将结果转回3D坐标 (n×3)
    # 对于纯旋转和平移变换（刚性变换），第四个分量始终为1，无需归一化
    transformed_points = transformed_points_homogeneous[:, :3]
    
    return transformed_points


def merge_point_clouds(points1, points2):
    """
    合并两个点云数据
    
    Args:
        points1: n×3的numpy数组，表示第一个点云数据，每行对应一个点的x、y、z坐标
        points2: m×3的numpy数组，表示第二个点云数据，每行对应一个点的x、y、z坐标
    
    Returns:
        numpy.ndarray: 合并后的点云数据，形状为((n+m)×3)
    """
    # 检查输入参数
    if points1.ndim != 2 or points1.shape[1] != 3:
        raise ValueError("第一个点云数据必须是n×3的数组")
    
    if points2.ndim != 2 or points2.shape[1] != 3:
        raise ValueError("第二个点云数据必须是m×3的数组")
    
    # 使用numpy的vstack函数高效合并两个点云数组
    # vstack是向量化操作，比循环或列表操作更高效
    merged_points = np.vstack([points1, points2])
    
    return merged_points


def save_points_to_ply(points, output_file, binary=False):
    """
    将点云数据保存为PLY格式文件
    
    Args:
        points: n×3的numpy数组，表示点云数据，每行对应一个点的x、y、z坐标
        output_file: 输出PLY文件路径
        binary: 是否以二进制格式保存，默认为ASCII格式
    
    Returns:
        bool: 保存是否成功
    """
    # 检查输入参数
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数据必须是n×3的数组")
    
    try:
        num_points = len(points)
        
        # 准备PLY文件头
        header = [
            'ply',
            'format ' + ('binary_little_endian' if binary else 'ascii') + ' 1.0',
            'element vertex ' + str(num_points),
            'property float x',
            'property float y',
            'property float z',
            'end_header'
        ]
        
        with open(output_file, 'wb' if binary else 'w') as f:
            # 写入文件头
            if binary:
                for line in header:
                    f.write((line + '\n').encode('ascii'))
                # 以二进制格式写入点云数据
                points.astype(np.float32).tofile(f)
            else:
                for line in header:
                    f.write(line + '\n')
                # 以ASCII格式写入点云数据
                # 使用numpy的向量化操作提高写入效率
                np.savetxt(f, points, fmt='%.6f %.6f %.6f')
        
        print(f"点云数据已成功保存到 {output_file}，共包含 {num_points} 个点")
        return True
    except Exception as e:
        print(f"保存点云数据到 {output_file} 时出错: {e}")
        return False


def generate_depth_map(points, resolution=1.0, output_file=None):
    """
    从点云数据生成深度图，使用xy作为像素坐标，z作为深度值
    
    Args:
        points: nx3的numpy数组，包含点云数据 [x, y, z]
        resolution: 像素分辨率（每个单位映射到多少像素），默认为1.0
        output_file: 输出深度图文件路径（可选）
    
    Returns:
        tuple: (depth_map, depth_image_normalized) - 原始深度图和归一化后的深度图像
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数据必须是n×3的数组")
    
    # 分离x, y, z坐标
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    
    # 计算点云在xy平面上的范围
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    # 计算图像尺寸
    width = int((x_max - x_min) * resolution + 1)
    height = int((y_max - y_min) * resolution + 1)
    
    print(f"深度图尺寸: {width}x{height} 像素")
    print(f"点云范围 - X: [{x_min:.2f}, {x_max:.2f}], Y: [{y_min:.2f}, {y_max:.2f}]")
    
    # 初始化深度图（使用较大的值表示无效深度）
    depth_map = np.full((height, width), np.inf)
    
    # 将点云映射到深度图
    for i in range(len(points)):
        x, y, z = points[i]
        
        # 计算像素坐标
        px = int((x - x_min) * resolution)
        py = int((y - y_min) * resolution)
        
        # 确保像素坐标在有效范围内
        if 0 <= px < width and 0 <= py < height:
            # 对于同一像素位置，保留最小的z值（通常表示最近的点）
            if z < depth_map[py, px]:
                depth_map[py, px] = z
    
    # 将无效深度（inf）设置为NaN
    depth_map[depth_map == np.inf] = np.nan
    
    # 归一化深度图用于可视化（0-255范围）
    valid_depth_mask = ~np.isnan(depth_map)
    if np.any(valid_depth_mask):
        # 提取有效深度值
        valid_depths = depth_map[valid_depth_mask]
        
        # 改进的前景背景分割方法
        # 1. 首先使用5%分位数过滤异常小的深度值
        lower_percentile = np.percentile(valid_depths, 1)
        toper_percentile = np.percentile(valid_depths, 100)
        
        # 2. 使用Otsu阈值法自动找到深度分布的最佳分割点
        # 将深度值离散化为直方图以便应用Otsu方法
        hist, bins = np.histogram(valid_depths[valid_depths >= lower_percentile], bins=518)
        
        # 实现增强型Otsu阈值算法，提高分割精度
        def otsu_threshold(histogram):
            # 1. 输入验证和预处理
            if len(histogram) <= 1:
                return 0
                
            # 移除直方图两端的零值，聚焦于有效区域
            non_zero_indices = np.where(histogram > 0)[0]
            if len(non_zero_indices) == 0:
                return 0
                
            start_idx, end_idx = non_zero_indices[0], non_zero_indices[-1]
            # 仅在非零区域内计算，提高精度并减少计算量
            trimmed_hist = histogram[start_idx:end_idx+1]
            
            total = trimmed_hist.sum()
            if total == 0:
                return 0
                
            # 2. 计算全局均值和累积值
            sum_total = np.dot(np.arange(len(trimmed_hist)), trimmed_hist)
            global_mean = sum_total / total
            
            # 3. 精确计算类间方差，寻找最佳阈值
            max_between_var = 0
            best_threshold = 0
            
            # 使用浮点数计算提高精度
            cumulative_weight = 0.0
            cumulative_mean = 0.0
            
            for i in range(len(trimmed_hist)):
                weight = trimmed_hist[i] / total  # 归一化权重
                cumulative_weight += weight
                
                if cumulative_weight <= 0 or cumulative_weight >= 1:
                    continue
                    
                mean = cumulative_mean * total / (total * cumulative_weight) if cumulative_weight > 0 else 0
                cumulative_mean += i * trimmed_hist[i] / total
                
                # 精确计算类间方差
                between_var = cumulative_weight * (1 - cumulative_weight) * ((cumulative_mean / cumulative_weight) - global_mean) ** 2
                
                # 4. 阈值细化：当方差相同时，选择更靠近中心的阈值
                if between_var > max_between_var or \
                   (np.isclose(between_var, max_between_var) and \
                    abs(i - len(trimmed_hist)/2) < abs(best_threshold - len(trimmed_hist)/2)):
                    max_between_var = between_var
                    best_threshold = i
            
            # 5. 映射回原始索引范围
            return start_idx + best_threshold
        
        # 计算最佳阈值并映射回原始深度范围
        if len(hist) > 1:  # 确保直方图有足够的数据点
            optimal_threshold_idx = otsu_threshold(hist)
            optimal_threshold = bins[optimal_threshold_idx]
            
            # 3. 结合分位数和Otsu结果，设置前景上限（优先考虑Otsu阈值，但添加保护机制）
            # 使用Otsu阈值作为主要分割点，但确保不会将前景范围设置得过大
            percentile_85 = np.percentile(valid_depths, 85)
            lower_percentile_foreground = min(optimal_threshold, percentile_85)
            
            print(f"深度分割：Otsu阈值={optimal_threshold:.4f}, 85%分位数={percentile_85:.4f}, 使用={lower_percentile_foreground:.4f}")

        
        # 4. 创建前景和背景掩码
        foreground_mask = (depth_map >= lower_percentile_foreground) & valid_depth_mask
        background_mask = valid_depth_mask & ~foreground_mask
        
        # 5. 统计信息
        foreground_ratio = np.sum(foreground_mask) / np.sum(valid_depth_mask) * 100 if np.sum(valid_depth_mask) > 0 else 0
        print(f"前景占比: {foreground_ratio:.1f}%, 前景深度范围: [{lower_percentile_foreground:.4f}, {toper_percentile:.4f}]")
        
        # 初始化彩色深度图为黑灰色
        depth_image_colored = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
        
        # 对前景区域应用红蓝彩色映射
        if np.any(foreground_mask):
            # 提取前景深度值
            foreground_depths = depth_map[foreground_mask]
            
            # 裁剪并归一化前景深度值
            clipped_foreground = np.clip(foreground_depths, lower_percentile_foreground, toper_percentile)
            normalized_foreground = (clipped_foreground - lower_percentile_foreground) / (toper_percentile - lower_percentile_foreground)
            
            # 使用伽马校正增强前景细节
            gamma = 0.7  # 更小的伽马值进一步增强前景对比度
            gamma_corrected = np.power(normalized_foreground, gamma)
            
            # 创建前景归一化图像
            foreground_normalized = np.zeros_like(depth_map, dtype=np.uint8)
            foreground_normalized[foreground_mask] = (255 * gamma_corrected).astype(np.uint8)
            
            # 增强前景细节
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            foreground_enhanced = clahe.apply(foreground_normalized) 
            
            # 应用JET色彩图到前景区域
            foreground_colored = cv2.applyColorMap(foreground_enhanced, cv2.COLORMAP_JET)
            
            # 将前景彩色区域复制到最终图像
            depth_image_colored[foreground_mask] = foreground_colored[foreground_mask]
        
        # 对背景区域（较远区域）应用简单的灰度映射（黑灰色）
        if np.any(background_mask):
            # 为背景区域创建灰度映射
            background_gray = np.zeros_like(depth_map, dtype=np.uint8)
            # 较低的灰度值，使远处呈现黑灰色
            background_gray[background_mask] = 30 + 50 * (depth_map[background_mask] - lower_percentile_foreground) / \
                                               (toper_percentile - lower_percentile_foreground)
            
            # 创建灰度的BGR表示
            background_gray_bgr = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
            
            # 将背景灰度区域复制到最终图像
            depth_image_colored[background_mask] = background_gray_bgr[background_mask]
        
        print(f"深度映射优化：使用Otsu自适应阈值法分割前景背景，前景应用红蓝彩色映射，远处区域显示为黑灰色")
    else:
        depth_image_colored = np.zeros((*depth_map.shape, 3), dtype=np.uint8)
    
    # 保存深度图（如果提供了输出路径）
    if output_file:
        # 保存彩色深度图像
        cv2.imwrite(output_file, depth_image_colored)
        print(f"彩色深度图已保存到: {output_file}")
        
        # 可选：保存原始深度值到npy文件
        npy_output = os.path.splitext(output_file)[0] + '_depth.npy'
        np.save(npy_output, depth_map)
        print(f"原始深度数据已保存到: {npy_output}")
    
    # 返回深度图和彩色深度图像
    return depth_map, depth_image_colored

def process_and_merge_point_clouds(input_file1, input_file2,transformation_matrix0, transformation_matrix1, uuid):
    """
    处理并合并点云数据，应用变换矩阵到第二个点云
    
    Args:
        input_file: 输入二进制文件路径
        transformation_matrix: 4×4的numpy数组变换矩阵
    
    Returns:
        numpy.ndarray: 合并后的点云数据，形状为(n+m)×3的numpy数组
    """
    # 检查变换矩阵参数
    if transformation_matrix0.shape != (4, 4):
        raise ValueError("变换矩阵必须是4×4的数组")
    if transformation_matrix1.shape != (4, 4):
        raise ValueError("变换矩阵必须是4×4的数组")
    
    # 1. 获取第一个点云数据
    print("正在获取第一个点云数据...")
    points1 = load_and_process_points(input_file1)
    """
    # 为第一个点云生成深度图
    print("\n为第一个点云生成深度图...")
    depth_map1, depth_image1 = generate_depth_map(points1, resolution=1.0)
    
    # 保存深度图示例 - 找不到路径时创建目录
    depth_output_dir = f"D:\\measure_result\\{uuid}"
    try:
        # 确保目录存在，如果不存在则创建
        os.makedirs(depth_output_dir, exist_ok=True)
        depth_output_file_0 = os.path.join(depth_output_dir, '0_depth.png')    
        cv2.imwrite(depth_output_file_0, depth_image1)
        print(f"第一个点云的深度图已保存到: {depth_output_file_0}")
    except Exception as e:
        print(f"保存深度图时出错: {e}")
    
    # 检查第一个点云是否为空
    if points1.size == 0:
        print("警告：第一个点云数据为空，无法继续处理")
        return np.array([])
      """
    # 2. 获取第二个点云数据
    print("\n正在获取第二个点云数据...")
    points2 = load_and_process_points(input_file2)
    
    # 检查第二个点云是否为空
    if points2.size == 0:
        print("警告：第二个点云数据为空，无法继续处理")
        return points1  # 返回第一个点云作为替代
    """
    # 为第二个点云生成深度图
    print("\n为第二个点云生成深度图...")
    depth_map2, depth_image2 = generate_depth_map(points2, resolution=1.0)
    
    # 保存深度图示例 - 找不到路径时直接放弃
    # depth_output_dir = f"C:\\measure_result\\{uuid}"
    if os.path.exists(depth_output_dir):
        depth_output_file_1 = os.path.join(depth_output_dir, '1_depth.png')    
        cv2.imwrite(depth_output_file_1, depth_image2)
        print(f"第二个点云的深度图已保存到: {depth_output_file_1}")
    else:
        pint(f"警告：路径 {depth_output_dir} 不存在，跳过深度图保存")
    """
    # 3. 对第二个点云应用坐标变换
    print("\n正在对第二个点云应用坐标变换...")
    transformed_points1 = transform_points(points1, transformation_matrix0)
    transformed_points2 = transform_points(points2, transformation_matrix1)
    
    # 4. 合并两个点云
    print("\n正在合并两个点云...")
    merged_points = merge_point_clouds(transformed_points1, transformed_points2)
    
    # 输出合并结果信息
    print(f"\n点云合并完成！")
    print(f"第一个点云点数: {len(points1)}")
    print(f"第二个点云点数: {len(points2)}")
    print(f"合并后的点云点数: {len(merged_points)}")
    print(f"合并后的点云形状: {merged_points.shape}")
    
    return merged_points


def adaptive_downsample(points):
    """
    对输入的点云数据进行自适应降采样
    
    Args:
        points: n×3的numpy数组，表示点云数据，每行对应一个点的x、y、z坐标
    
    Returns:
        numpy.ndarray: 降采样后的点云数据，形状为(m×3)，其中m ≤ n
    """
    # 检查输入参数
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("点云数据必须是n×3的数组")
    
    # 计算点云的包围盒对角线长度
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    diagonal_length = np.linalg.norm(max_coords - min_coords)
    print(f"点云包围盒对角线长度: {diagonal_length:.4f}")
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    processed_pcd = pcd
    
    # 根据点云规模调整体素大小，确保保留足够的点
    num_points = len(points)
    if num_points > 100000:
        voxel_size = diagonal_length * 0.0005  # 减小体素大小，保留更多点
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
    
    # 将Open3D点云对象转换回numpy数组
    # downsampled_points = np.asarray(processed_pcd.points)
    
    
    return processed_pcd


def extract_largest_cluster(pcd,distance_threshold= 10.0,min_points_per_cluster=30):
    """
    基于距离的聚类算法，筛选出最大的聚类
    
    Args:
        pcd: Open3D点云对象
    
    Returns:
        Open3D点云对象: 仅包含最大聚类的点云
    """
    # 检查输入参数
    if not isinstance(pcd, o3d.geometry.PointCloud):
        raise ValueError("输入必须是Open3D点云对象")
    
    # 检查点云是否为空
    if len(pcd.points) == 0:
        print("警告：输入点云为空")
        return pcd
    
    print(f"\n执行基于距离的聚类分析...")
    print(f"原始点云点数: {len(pcd.points)}")
    
    # # 设置聚类参数
    # distance_threshold = 10.0  # 距离阈值
    # min_points_per_cluster = 30  # 每个聚类的最小点数
    
    # 执行DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps=distance_threshold, min_points=min_points_per_cluster, print_progress=False))
    
    # 计算每个聚类的点数
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 过滤出符合条件的聚类（点数≥30）
    valid_clusters = []
    for label, count in zip(unique_labels, counts):
        # 忽略噪声点（label=-1）
        if label != -1 and count >= min_points_per_cluster:
            valid_clusters.append((label, count))
    
    print(f"检测到的聚类总数: {len(unique_labels)}")
    print(f"有效聚类数量（点数≥{min_points_per_cluster}）: {len(valid_clusters)}")
    
    if not valid_clusters:
        print("警告：未找到符合条件的聚类")
        return o3d.geometry.PointCloud()
    
    # 找出最大的聚类
    largest_cluster = max(valid_clusters, key=lambda x: x[1])
    largest_label, largest_count = largest_cluster
    
    print(f"最大聚类标签: {largest_label}, 点数: {largest_count}")
    print(f"聚类占原始点云比例: {largest_count / len(pcd.points) * 100:.2f}%")
    
    # 提取最大聚类的点
    points_array = np.asarray(pcd.points)
    largest_cluster_indices = labels == largest_label
    largest_cluster_points = points_array[largest_cluster_indices]
    
    # 创建新的点云对象
    result_pcd = o3d.geometry.PointCloud()
    result_pcd.points = o3d.utility.Vector3dVector(largest_cluster_points)
    
    print(f"提取最大聚类后的点云点数: {len(result_pcd.points)}")
    
    return result_pcd





def main_point_cloud_processing(input_file1, input_file2, transformation_matrix=None, 
                               save_downsampled=True, save_cluster=True, 
                               outputdir='measurements_results',
                               downsampled_filename='01下采样.ply', 
                               cluster_filename='02聚类去噪.ply',
                               uuid=None
                               ):
    """
    点云处理主函数：合并、降采样和聚类分析
    
    Args:
        input_file1: str - 第一个点云数据文件路径
        input_file2: str - 第二个点云数据文件路径
        transformation_matrix: np.ndarray, optional - 4×4变换矩阵，默认使用预定义矩阵
        save_downsampled: bool, optional - 是否保存降采样后的点云，默认为True
        save_cluster: bool, optional - 是否保存最大聚类的点云，默认为True
        outputdir: str, optional - 输出文件夹路径，默认为'measurements_results'
        downsampled_filename: str, optional - 降采样点云文件名
        cluster_filename: str, optional - 最大聚类点云文件名
    
    Returns:
        dict: 包含处理结果的字典，具有以下键：
            - merged_points: np.ndarray - 合并后的点云数据
            - downsampled_pcd: o3d.geometry.PointCloud - 降采样后的点云对象
            - largest_cluster_pcd: o3d.geometry.PointCloud - 最大聚类的点云对象
            - downsampled_saved: bool - 降采样点云是否成功保存
            - cluster_saved: bool - 最大聚类点云是否成功保存
    """
    # 创建输出文件夹
    os.makedirs(outputdir, exist_ok=True)
    print(f"已创建输出文件夹: {outputdir}")
    
    # 构建完整的输出文件路径
    downsampled_output = os.path.join(outputdir, downsampled_filename)
    cluster_output = os.path.join(outputdir, cluster_filename)
    
    # 如果未提供变换矩阵，使用默认矩阵

    # 根据用户提供的旋转矩阵、缩放因子和平移向量构建4x4变换矩阵
    # 不旋转的变换矩阵（单位矩阵）
    transformation_matrix1 = np.array([
        [1.0, 0.0, 0.0, 0.0],  # [R11, R12, R13, Tx] - 单位矩阵，无旋转无平移
        [0.0, 1.0, 0.0, 0.0],  # [R21, R22, R23, Ty]
        [0.0, 0.0, 1.0, 0.0],  # [R31, R32, R33, Tz]
        [0.0, 0.0, 0.0, 1.0]   # [0, 0, 0, 1]
    ])
    
    # 根据用户提供的旋转矩阵、缩放因子和平移向量构建4x4变换矩阵
    # 使用用户提供的更精确的变换矩阵
    transformation_matrix0 = np.array([
        [-0.152911025482, 0.000130377867, -0.988239951271, 638.406442624892],    # [R11, R12, R13, Tx]
        [-0.000121558653, 0.999999981251, 0.000150738211, -14.585486261261],      # [R21, R22, R23, Ty]
        [-0.988239952395, -0.000143178652, 0.152911006767, -205.016362061633],    # [R31, R32, R33, Tz]
        [0.000000000000, 0.000000000000, 0.000000000000, 1.000000000000]         # [0, 0, 0, 1]
    ])
    
    
    # 初始化结果字典
    result = {
        'merged_points': None,
        'downsampled_pcd': None,
        'largest_cluster_pcd': None,
        'downsampled_saved': False,
        'cluster_saved': False
    }
    
    print(f"输入文件1: {input_file1}")
    print(f"输入文件2: {input_file2}")
    print("\n使用的变换矩阵:")
    print(transformation_matrix0, transformation_matrix1)
    
    # 检查文件是否存在
    if not os.path.exists(input_file1):
        print(f"错误: 文件 {input_file1} 不存在!")
        return result
    
    if not os.path.exists(input_file2):
        print(f"错误: 文件 {input_file2} 不存在!")
        return result
    
    try:
        print("\n=== 开始点云处理流程 ===")
        
        # 1. 处理并合并点云
        print("\n处理并合并点云数据...")
        merged_points = process_and_merge_point_clouds(input_file1, input_file2, transformation_matrix0, transformation_matrix1, uuid=uuid)
        result['merged_points'] = merged_points
        
        if len(merged_points) == 0:
            print("处理失败，未获取到合并后的点云数据。")
            return result
        
        print("\n点云处理和合并成功！")
        
        # 2. 自适应降采样
        print("\n=== 执行自适应降采样 ===")
        print(f"原始合并点云点数: {len(merged_points)}")
        # voxel_size = diagonal_length * 0.0005
        downsampled_pcd = adaptive_downsample(merged_points)
        result['downsampled_pcd'] = downsampled_pcd
        
        print(f"降采样后的点云点数: {len(downsampled_pcd.points)}")
        print(f"降采样率: {(1 - len(downsampled_pcd.points) / len(merged_points)) * 100:.2f}%")
        
        # 3. 基于距离的聚类算法
        print("\n=== 执行基于距离的聚类算法 ===")
        largest_cluster_pcd = extract_largest_cluster(downsampled_pcd)
        result['largest_cluster_pcd'] = largest_cluster_pcd

        print("\n=== 执行二次聚类 ===")
        largest_cluster_pcd2 = extract_largest_cluster(largest_cluster_pcd,distance_threshold= 2, min_points_per_cluster=30)
        result['largest_cluster_pcd2'] = largest_cluster_pcd2
        
        # 4. 保存降采样后的点云
        if save_downsampled:
            print("\n保存降采样后的点云到PLY文件...")
            downsampled_points_np = np.asarray(downsampled_pcd.points)
            downsampled_result = save_points_to_ply(downsampled_points_np, downsampled_output, binary=True)
            result['downsampled_saved'] = downsampled_result
            print(f"降采样点云文件保存{'成功' if downsampled_result else '失败'}: {downsampled_output}")
        
        # 5. 保存最大聚类的点云
        if save_cluster and len(largest_cluster_pcd.points) > 0:
            print("\n保存最大聚类点云到PLY文件...")
            largest_cluster_points_np = np.asarray(largest_cluster_pcd.points)
            cluster_result = save_points_to_ply(largest_cluster_points_np, cluster_output, binary=True)
            result['cluster_saved'] = cluster_result
            print(f"最大聚类点云文件保存{'成功' if cluster_result else '失败'}: {cluster_output}")

                # 5. 保存最大聚类的点云
        if save_cluster and len(largest_cluster_pcd2.points) > 0:
            print("\n保存最大聚类点云到PLY文件...")
            largest_cluster_points_np2 = np.asarray(largest_cluster_pcd2.points)
            # 修正：定义cluster_output2变量
            cluster_output2 = os.path.join(outputdir, '03二次聚类.ply')
            cluster_result2 = save_points_to_ply(largest_cluster_points_np2, cluster_output2, binary=True)
            result['cluster_2saved'] = cluster_result2
            print(f"最大聚类点云文件保存{'成功' if cluster_result2 else '失败'}: {cluster_output2}")
        
        print("\n=== 点云处理流程完成 ===")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
    
    return result


if __name__ == "__main__":
    # 示例使用方式
    default_file1 = r"c:/Users/30583/Desktop/1106上午测试数据/pointData-0-f30ed67b-07e7-456d-b417-b5bf03568f35.txt"
    default_file2 = r"C:/Users/30583/Desktop/1106上午测试数据/pointData-1-f30ed67b-07e7-456d-b417-b5bf03568f35.txt"
    
    # 调用主函数进行点云处理
    processing_result = main_point_cloud_processing(
        input_file1=default_file1,
        input_file2=default_file2,
        save_downsampled=True,
        save_cluster=True,
        outputdir='measurements_results',
        downsampled_filename='01下采样.ply',
        cluster_filename='02聚类去噪.ply'
    )
    
    # 显示处理结果摘要
    print("\n=== 处理结果摘要 ===")
    if processing_result['merged_points'] is not None:
        print(f"合并点云点数: {len(processing_result['merged_points'])}")
    if processing_result['downsampled_pcd'] is not None:
        print(f"降采样后点数: {len(processing_result['downsampled_pcd'].points)}")
    if processing_result['largest_cluster_pcd'] is not None:
        print(f"最大聚类点数: {len(processing_result['largest_cluster_pcd'].points)}")
    print(f"降采样文件保存: {'成功' if processing_result['downsampled_saved'] else '失败'}")
    print(f"聚类文件保存: {'成功' if processing_result['cluster_saved'] else '失败'}")
    
