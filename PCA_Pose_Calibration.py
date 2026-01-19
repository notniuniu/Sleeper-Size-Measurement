import open3d as o3d
import numpy as np
import copy

def advanced_pca_with_reference(source_pcd, output_pcd_path,reference_pcd_path="./参考.ply",):
    """
    使用PCA粗配准结合参考模型进行精确对齐
    
    参数:
        input_pcd_path: 输入待对齐点云路径
        reference_pcd_path: 参考模型点云路径（已对齐的理想模型）
        output_pcd_path: 输出对齐后的点云路径
        known_dimensions: 已知物理尺寸（可选）
    """
    # 1. 读取点云
    # source_pcd = input_pcd
    target_pcd = o3d.io.read_point_cloud(reference_pcd_path)
    
    if len(source_pcd.points) == 0 or len(target_pcd.points) == 0:
        raise ValueError("点云文件读取失败或为空")
    
    print(f"源点云点数: {len(source_pcd.points)}")
    print(f"目标点云点数: {len(target_pcd.points)}")
    
    # # 2. 预处理：去噪和下采样
    # source_pcd = source_pcd.voxel_down_sample(voxel_size=5.0)
    # target_pcd = target_pcd.voxel_down_sample(voxel_size=5.0)
    
    # 3. 计算法向量（为后续ICP准备）
    source_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    target_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
    
    # # 4. 基于物理尺寸的PCA粗配准
    # def pca_coarse_alignment(pcd, known_dims=None):
    #     points = np.asarray(pcd.points)
    #     centroid = np.mean(points, axis=0)
    #     centered_points = points - centroid
        
    #     # PCA计算
    #     covariance_matrix = np.cov(centered_points, rowvar=False)
    #     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
    #     # 按特征值排序
    #     sorted_indices = np.argsort(eigenvalues)[::-1]
    #     eigenvectors_sorted = eigenvectors[:, sorted_indices]
        
    #     # 如果提供已知尺寸，使用尺寸约束校正
    #     if known_dims is not None:
    #         # 计算各主方向上的投影跨度
    #         projected_ranges = []
    #         for i in range(3):
    #             projection = np.dot(centered_points, eigenvectors_sorted[:, i])
    #             proj_range = np.max(projection) - np.min(projection)
    #             projected_ranges.append(proj_range)
            
    #         # 按跨度大小排序
    #         range_order = np.argsort(projected_ranges)[::-1]
            
    #         # 重新排列特征向量
    #         corrected_vectors = np.zeros((3, 3))
    #         corrected_vectors[:, 0] = eigenvectors_sorted[:, range_order[0]]  # 最长方向->X
    #         corrected_vectors[:, 1] = eigenvectors_sorted[:, range_order[1]]  # 次长方向->Y
    #         corrected_vectors[:, 2] = np.cross(corrected_vectors[:, 0], corrected_vectors[:, 1])
            
    #         # 方向校正
    #         for i in range(3):
    #             if np.sum(corrected_vectors[:, i]) < 0:
    #                 corrected_vectors[:, i] = -corrected_vectors[:, i]
            
    #         rotation_matrix = corrected_vectors.T
    #     else:
    #         rotation_matrix = eigenvectors_sorted.T
        
    #     # 构建变换矩阵
    #     transform = np.eye(4)
    #     transform[:3, :3] = rotation_matrix
    #     transform[:3, 3] = -rotation_matrix @ centroid
        
    #     return transform
    
    # # 应用PCA粗配准
    # coarse_transform = pca_coarse_alignment(source_pcd, known_dimensions)
    # source_coarse = copy.deepcopy(source_pcd)
    # source_coarse.transform(coarse_transform)
    
    # print("PCA粗配准完成")
    
    # 5. ICP精配准
    def refined_icp_registration(source, target, max_correspondence_distance=50.0):
        """
        改进的ICP配准，结合多种策略提高精度
        """
        # 策略1：点对点ICP
        icp_p2p = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        # 策略2：点对面ICP（通常更精确）
        icp_p2plane = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        # 选择更好的结果
        if icp_p2plane.fitness > icp_p2p.fitness:
            return icp_p2plane.transformation, icp_p2plane
        else:
            return icp_p2p.transformation, icp_p2p
    
    # 执行ICP精配准
    icp_transform, icp_result = refined_icp_registration(
        source_pcd, target_pcd, max_correspondence_distance=100.0
    )
    
    print(f"ICP配准结果 - 适应度: {icp_result.fitness:.4f}, RMSE: {icp_result.inlier_rmse:.4f}")
    
    # 6. 组合变换并应用
    final_transform = icp_transform  # 注意变换矩阵的组合顺序
    source_aligned = copy.deepcopy(source_pcd)
    source_aligned.transform(final_transform)
    
    # 7. 保存结果
    o3d.io.write_point_cloud(output_pcd_path, source_aligned)
    
    # 8. 评估配准质量
    def evaluate_registration(source, target):
        """评估配准质量"""
        # 计算配准误差
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source, target, max_correspondence_distance=50.0
        )
        
        # 计算点云边界框尺寸
        source_bbox = source.get_axis_aligned_bounding_box()
        target_bbox = target.get_axis_aligned_bounding_box()
        
        source_dims = source_bbox.get_extent()
        target_dims = target_bbox.get_extent()
        
        print("=== 配准质量评估 ===")
        print(f"配准适应度: {evaluation.fitness:.4f}")
        print(f"内点RMSE: {evaluation.inlier_rmse:.4f}")
        print(f"源点云尺寸: {source_dims}")
        print(f"目标点云尺寸: {target_dims}")
        print(f"尺寸差异: {np.abs(source_dims - target_dims)}")
        
        return evaluation
    
    evaluation_result = evaluate_registration(source_aligned, target_pcd)
    
    return source_aligned, final_transform, evaluation_result

def visualize_registration_results(original_pcd, coarse_pcd, aligned_pcd, target_pcd):
    """可视化配准过程各阶段结果"""
    # 着色
    original_pcd.paint_uniform_color([1, 0, 0])    # 红色-原始
    coarse_pcd.paint_uniform_color([0, 1, 0])     # 绿色-粗配准后
    aligned_pcd.paint_uniform_color([0, 0, 1])   # 蓝色-精配准后
    target_pcd.paint_uniform_color([1, 1, 0])    # 黄色-参考模型
    
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500, origin=[0, 0, 0])
    
    # 分视图显示
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="点云配准全过程", width=1200, height=800)
    
    # 添加点云到不同视图
    vis.add_geometry(original_pcd)
    vis.add_geometry(coarse_pcd)
    vis.add_geometry(aligned_pcd)
    vis.add_geometry(target_pcd)
    vis.add_geometry(coordinate_frame)
    
    # 设置渲染选项
    opt = vis.get_render_option()
    opt.point_size = 2.0
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    # 使用示例
    input_file = "results/1552efbf-754b-444b-9113-3d7c1cbb2ffa/03预旋转.ply"           # 待对齐点云
    reference_file = "参考.ply"   # 理想参考模型
    output_file = "final_aligned.pcd"  # 输出文件
    
    # 已知物理尺寸（可选）
    # 定义known变量为True表示使用已知的物理尺寸
    # known = True
    # physical_dims = [2771.55, 539.821, 476.266] if known else None
    
    try:
        # 执行配准
        aligned_pcd, transform, evaluation = advanced_pca_with_reference(
            input_file, reference_file, output_file
        )
        
        # 可视化（可选）
        # original_pcd = o3d.io.read_point_cloud(input_file)
        # reference_pcd = o3d.io.read_point_cloud(reference_file)
        
        # # 需要保存粗配准结果用于可视化
        # coarse_pcd = copy.deepcopy(original_pcd)
        # # coarse_transform = ...  # 这里需要从函数中返回粗配准结果
        
        # visualize_registration_results(original_pcd, coarse_pcd, aligned_pcd, reference_pcd)
        
    except Exception as e:
        print(f"处理错误: {e}")