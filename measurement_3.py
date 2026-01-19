import os
import numpy as np
import open3d as o3d
import logging
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, MultiPoint
import matplotlib.pyplot as plt
from parallelogram_measurement import calculate_parallelogram_length_width

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlockMeasurement:
    """
    立体梯形单元块点云测量类
    用于计算单元块的上宽W0、下宽W1、上长L0和下长L1等几何参数
    使用分层切片法进行精确测量
    """
    
    def __init__(self, output_dir="test_results"):
        """
        初始化测量器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置算法参数
        self.k_neighbors = 50  # 统计离群点移除的邻域点数
        self.std_ratio = 1.0   # 统计离群点移除的标准差倍数
        self.top_slice_ratio = 0.15  # 顶部切片区域比例（5%）
        self.bottom_slice_ratio = 0.5  # 底部切片区域比例（5%）
        self.top_slice_count = 1  # 顶部区域切片数量
        self.bottom_slice_count = 1  # 底部区域切片数量
        self.alpha = 10 # Alpha形状算法的alpha参数
    
    def _remove_outliers(self, pcd):
        """
        使用统计学方法去除离群点
        
        Args:
            pcd: 输入点云
            
        Returns:
            去噪后的点云
        """
        logger.info(f"开始去除离群点，原始点云包含{len(pcd.points)}个点")
        
        # 使用Open3D的统计离群点移除
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=self.k_neighbors,
            std_ratio=self.std_ratio
        )
        
        inlier_cloud = pcd.select_by_index(ind)
        outlier_cloud = pcd.select_by_index(ind, invert=True)
        
        logger.info(f"离群点移除完成: 保留{len(inlier_cloud.points)}个点，移除{len(outlier_cloud.points)}个离群点")
        
        # 保存去噪结果
        self._save_point_cloud(inlier_cloud, "单元块_去噪.ply", [0, 1, 0])
        
        return inlier_cloud
    
    def _generate_slices(self, pcd):
        """
        生成Z轴方向的切片（改进版：只返回顶部和底部的Z值范围）
        
        Args:
            pcd: 输入点云
            
        Returns:
            (顶部Z值范围, 底部Z值范围)
        """
        points = np.asarray(pcd.points)
        z_values = points[:, 2]
        
        # 计算Z值范围
        min_z = np.min(z_values)
        max_z = np.max(z_values)
        z_range = max_z - min_z
        
        logger.info(f"Z值范围: [{min_z:.4f}, {max_z:.4f}], 范围长度: {z_range:.4f}")
        
        # 直接返回顶部和底部的Z值，不再生成多个切片
        # 使用整个顶部和底部区域，不再使用比例
        return max_z, min_z
    
    def _extract_slice_contour(self, pcd, z_value, tolerance_percentage=0.1, is_top=True):
        """
        提取特定Z值附近的所有点作为切片，使用基于点云Z轴范围百分比的容差
        
        Args:
            pcd: 输入点云
            z_value: 切片Z值（最高或最低Z值）
            tolerance_percentage: Z值容差百分比（占Z轴范围的比例，默认5%）
            is_top: 是否为顶部切片
            
        Returns:
            二维轮廓点集 (n, 2) 和 切片点云
        """
        points = np.asarray(pcd.points)
        
        # 计算点云在Z轴方向的范围
        z_min = np.min(points[:, 2])
        z_max = np.max(points[:, 2])
        z_range = z_max - z_min
        
        # 根据Z轴范围的百分比来确定容差值
        tolerance = z_range * tolerance_percentage
        logger.info(f"计算容差值: Z范围={z_range:.4f}, 容差百分比={tolerance_percentage:.2%}, 实际容差={tolerance:.4f}")
        
        # 使用计算出的容差提取Z值附近的所有点
        if is_top:
            # 顶部切片：Z值大于等于 (z_value - tolerance)
            mask = points[:, 2] >= (z_value - tolerance)
        else:
            # 底部切片：Z值小于等于 (z_value + tolerance)
            mask = points[:, 2] <= (z_value + tolerance)
        
        slice_points = points[mask]
        
        if len(slice_points) == 0:
            logger.error(f"切片Z={z_value:.4f} 未能提取到足够的点，尝试容差={tolerance:.4f}")
            return np.array([]), o3d.geometry.PointCloud()
        
        logger.info(f"{'顶部' if is_top else '底部'}切片Z={z_value:.4f} 提取到{len(slice_points)}个点，容差: {tolerance:.4f}")
        
        # 创建切片点云
        slice_pcd = o3d.geometry.PointCloud()
        slice_pcd.points = o3d.utility.Vector3dVector(slice_points)
        
        # 投影到XY平面
        xy_points = slice_points[:, :2]
        
        return xy_points, slice_pcd
    


    def _compute_contour_polygon(self, xy_points):
        """
        使用Alpha Shape算法计算二维点集的外轮廓，保留更多的轮廓点

        Args:
            xy_points: 二维点集 (n, 2)
            
        Returns:
            轮廓顶点 (m, 2)，按顺时针顺序排列
        """
        if len(xy_points) < 3:
            return xy_points

        # 获取alpha参数，默认为0.1
        alpha = getattr(self, 'alpha', 200)

        try:
            # 方法1: 使用Delaunay三角剖分实现Alpha Shape
            return self._alpha_shape_delaunay(xy_points, alpha)
        except Exception as e:
            print(f"Alpha Shape Delaunay方法失败: {e}, 使用凸包作为备选")
            # 备选方案：凸包算法
            return self._convex_hull_fallback(xy_points)

    def _alpha_shape_delaunay(self, points, alpha):
        """
        基于Delaunay三角剖分的Alpha Shape实现[8,12](@ref)

        Args:
            points: 二维点坐标数组 (n, 2)
            alpha: Alpha形状参数，控制轮廓的精细程度
            
        Returns:
            轮廓点坐标数组 (m, 2)
        """
        if len(points) < 4:
            return self._convex_hull_fallback(points)
        
        # 构建Delaunay三角剖分
        tri = Delaunay(points)
        
        # 统计每条边的出现次数
        edge_count = {}
        
        # 遍历每个三角形
        for simplex in tri.simplices:
            # 获取三角形的三条边
            edges = [
                (min(simplex[0], simplex[1]), max(simplex[0], simplex[1])),
                (min(simplex[1], simplex[2]), max(simplex[1], simplex[2])),
                (min(simplex[2], simplex[0]), max(simplex[2], simplex[0]))
            ]
            
            # 计算外接圆半径
            p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            a = np.linalg.norm(p2 - p3)
            b = np.linalg.norm(p1 - p3)
            c = np.linalg.norm(p1 - p2)
            
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
            circum_radius = a * b * c / (4 * area) if area > 0 else float('inf')
            
            # 如果外接圆半径小于alpha，记录这些边
            if circum_radius < alpha:
                for edge in edges:
                    edge_count[edge] = edge_count.get(edge, 0) + 1
        
        # 边界边是只被一个三角形包含的边（出现次数为1）
        boundary_edges = [edge for edge, count in edge_count.items() if count == 1]
        if not edges:
            # 如果没有找到合适的边，使用凸包
            return self._convex_hull_fallback(points)

        # 构建边界多边形
        boundary_points = self._build_boundary_from_edges(points, boundary_edges)

        return boundary_points

    def _build_boundary_from_edges(self, points, edges):
        """
        从边界边构建连续的多边形轮廓[7](@ref)

        Args:
            points: 所有点坐标
            edges: 边界边集合
            
        Returns:
            有序的边界点坐标
        """
        if not edges:
            return points

        # 构建邻接表
        graph = {}
        for i, j in edges:
            if i not in graph:
                graph[i] = []
            if j not in graph:
                graph[j] = []
            graph[i].append(j)
            graph[j].append(i)

        # 找到边界环
        boundaries = []
        visited = set()

        for start_node in graph:
            if start_node in visited or len(graph[start_node]) == 0:
                continue
                
            # 深度优先搜索寻找闭环
            stack = [start_node]
            current_path = []
            
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                    
                visited.add(node)
                current_path.append(node)
                
                # 按角度排序邻居节点，保持顺时针方向
                if node in graph and graph[node]:
                    neighbors = graph[node]
                    if len(current_path) > 1:
                        # 计算方向向量，保持顺时针
                        prev_point = points[current_path[-2]]
                        current_point = points[node]
                        direction = current_point - prev_point
                        
                        # 按相对于当前方向的角度排序邻居
                        neighbors.sort(key=lambda n: self._angle_between_points(
                            current_point, points[n], direction))
                    
                    stack.extend(neighbors)
            
            if len(current_path) > 2:
                boundaries.append(current_path)

        # 选择最长的边界作为外轮廓
        if boundaries:
            main_boundary = max(boundaries, key=len)
            boundary_points = points[main_boundary]
            
            # 确保点集是闭合的
            if not np.array_equal(boundary_points[0], boundary_points[-1]):
                boundary_points = np.vstack([boundary_points, boundary_points[0:1]])
            
            return boundary_points
        else:
            return self._convex_hull_fallback(points)

    def _angle_between_points(self, center, target, reference_direction):
        """
        计算从参考方向到目标点的角度[7](@ref)
        """
        vector = target - center
        angle = np.arctan2(vector[1], vector[0]) - np.arctan2(reference_direction[1], reference_direction[0])
        return angle % (2 * np.pi)

    def _convex_hull_fallback(self, points):
        """
        凸包备选算法[13](@ref)

        Args:
            points: 二维点坐标数组
            
        Returns:
            凸包点坐标数组
        """
        from scipy.spatial import ConvexHull

        if len(points) < 3:
            return points

        hull = ConvexHull(points)
        hull_points = points[hull.vertices]

        # 确保多边形闭合
        if not np.array_equal(hull_points[0], hull_points[-1]):
            hull_points = np.vstack([hull_points, hull_points[0:1]])

        return hull_points

    def _alpha_shape_brute_force(self, points, alpha):
        """
        暴力实现的Alpha Shape算法（适用于小数据集）[6,7](@ref)

        Args:
            points: 点坐标数组
            alpha: alpha参数
            
        Returns:
            边界点坐标
        """
        n = len(points)
        edges = set()

        for i in range(n):
            for j in range(i + 1, n):
                p1, p2 = points[i], points[j]
                distance = np.linalg.norm(p1 - p2)
                
                if distance > 2 * alpha:
                    continue
                    
                # 计算过两点的圆的圆心[7](@ref)
                mid_point = (p1 + p2) / 2
                perpendicular = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                perpendicular = perpendicular / np.linalg.norm(perpendicular)
                
                height = np.sqrt(alpha**2 - (distance/2)**2)
                center1 = mid_point + height * perpendicular
                center2 = mid_point - height * perpendicular
                
                # 检查圆内是否没有其他点
                valid_edge = True
                for k in range(n):
                    if k == i or k == j:
                        continue
                        
                    pk = points[k]
                    dist1 = np.linalg.norm(pk - center1)
                    dist2 = np.linalg.norm(pk - center2)
                    
                    if dist1 < alpha or dist2 < alpha:
                        valid_edge = False
                        break
                
                if valid_edge:
                    edges.add((i, j))

        return self._build_boundary_from_edges(points, edges)
    
    
    def _save_contour(self, contour_points, z_value, filename, color=[0, 0, 1]):
        """
        保存轮廓点到PLY文件，将二维轮廓点反向投影回三维空间
        
        Args:
            contour_points: 二维轮廓点集 (n, 2)
            z_value: 对应的Z值（取原始点集的极值）
            filename: 文件名
            color: RGB颜色
        """
        try:
            if len(contour_points) == 0:
                logger.warning(f"空轮廓，跳过保存: {filename}")
                return
            
            # 将二维点转换为三维点（Z值设为切片的极值Z值）
            three_d_points = np.hstack([contour_points, np.full((len(contour_points), 1), z_value)])
            
            # 创建点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(three_d_points)
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(contour_points), 1)))
            
            # 保存文件
            save_path = os.path.join(self.output_dir, filename)
            o3d.io.write_point_cloud(save_path, pcd)
            logger.info(f"轮廓保存成功: {save_path}, 包含{len(contour_points)}个点")
        except Exception as e:
            logger.error(f"轮廓保存失败 {filename}: {e}")
    
    def _save_point_cloud(self, pcd, filename, color=[1, 0, 0]):
        """
        保存点云到文件
        
        Args:
            pcd: 点云对象
            filename: 文件名
            color: RGB颜色
        """
        try:
            if len(pcd.points) == 0:
                logger.warning(f"空点云，跳过保存: {filename}")
                return
            
            # 设置颜色
            colored_pcd = o3d.geometry.PointCloud(pcd)
            colored_pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(colored_pcd.points), 1)))
            
            # 保存文件
            save_path = os.path.join(self.output_dir, filename)
            o3d.io.write_point_cloud(save_path, colored_pcd)
            logger.info(f"点云保存成功: {save_path}, 包含{len(pcd.points)}个点")
        except Exception as e:
            logger.error(f"点云保存失败 {filename}: {e}")
    
    def _calculate_dimensions_from_contours(self, contours):
        """
        从轮廓点集计算尺寸
        
        Args:
            contours: 轮廓点集列表
            
        Returns:
            (宽度列表, 长度列表)
        """
        widths = []  # Y轴方向的宽度
        lengths = []  # X轴方向的长度
        
        for i, contour in enumerate(contours):
            if len(contour) < 3:
                logger.warning(f"轮廓{i}点太少，跳过")
                continue
            
            # 计算宽度（Y轴方向）
            y_min = np.min(contour[:, 1])
            y_max = np.max(contour[:, 1])
            width = y_max - y_min
            widths.append(width)
            
            # 计算长度（X轴方向）
            x_min = np.min(contour[:, 0])
            x_max = np.max(contour[:, 0])
            length = x_max - x_min
            lengths.append(length)
            
            logger.info(f"轮廓{i}: 宽度={width:.4f}, 长度={length:.4f}")
        
        return widths, lengths

    def _get_bounding_box_dimensions(self, pcd):
        """
        计算点云的长度和宽度，基于距离聚类、直线拟合和范围筛选
        
        Args:
            pcd: Open3D点云对象
            
        Returns:
            tuple: (长度, 宽度)，其中长度对应X轴方向，宽度基于两条拟合直线间的距离
        """
        import numpy as np
        from sklearn.cluster import DBSCAN
        from sklearn.linear_model import LinearRegression
        import logging
        
        # 获取logger
        logger = logging.getLogger(__name__)
        
        # 检查点云是否为空
        if len(pcd.points) == 0:
            raise ValueError("输入的点云为空")
        
        # 获取点云坐标
        points = np.asarray(pcd.points)
        
        # 步骤1: 使用距离聚类算法将点云分为2个独立的簇
        logger.info(f"开始聚类分析，共有{len(points)}个点")
        
        # 使用DBSCAN进行距离聚类
        # 动态设置eps参数，基于点云分布自适应调整
        # 计算点云在Y方向的范围作为参考
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        eps = y_range * 0.1  # 设置为Y轴范围的10%
        
        # 执行聚类
        clustering = DBSCAN(eps=eps, min_samples=5).fit(points)
        labels = clustering.labels_
        
        # 统计有效簇的数量（排除噪声点标签-1）
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 移除噪声点标签
        
        # 如果聚类结果不是正好2个簇，使用k-means作为备选方案
        if len(unique_labels) != 2:
            logger.warning(f"DBSCAN聚类结果为{len(unique_labels)}个簇，尝试使用k-means聚类")
            from sklearn.cluster import KMeans
            # 使用k-means强制分为2个簇
            kmeans = KMeans(n_clusters=2, random_state=42).fit(points[:, 1].reshape(-1, 1))  # 基于Y坐标聚类
            labels = kmeans.labels_
            unique_labels = {0, 1}
        
        logger.info(f"成功获取2个点簇")
        
        # 步骤2: 对每个簇拟合平行于X轴的直线
        fitted_lines = []
        cluster_points_list = []
        
        for label in unique_labels:
            # 获取当前簇的点
            cluster_points = points[labels == label]
            cluster_points_list.append(cluster_points)
            
            # 提取X和Y坐标用于拟合
            X = cluster_points[:, 0].reshape(-1, 1)  # 自变量为X坐标
            y = cluster_points[:, 1]  # 因变量为Y坐标
            
            # 拟合直线 (Y = a*X + b，但由于要平行于X轴，实际上我们只需要Y的平均值)
            # 对于平行于X轴的直线，Y值基本不变，所以使用线性回归并约束斜率为0
            # 这里我们直接计算Y值的平均值作为直线位置
            line_y = np.mean(y)
            fitted_lines.append(line_y)
            
            logger.info(f"簇{label}拟合直线Y={line_y:.4f}")
        
        # 步骤3: 计算两条拟合直线之间的垂直距离作为宽度
        width = abs(fitted_lines[0] - fitted_lines[1])
        logger.info(f"计算得到宽度: {width:.4f}")
        
        # 步骤4: 筛选位于两条直线范围内的点云数据
        y_min_line = min(fitted_lines)
        y_max_line = max(fitted_lines)
        
        # 筛选Y坐标在两条直线范围内的点
        filtered_points = points[(points[:, 1] >= y_min_line) & (points[:, 1] <= y_max_line)]
        
        if len(filtered_points) == 0:
            logger.warning("未找到位于直线范围内的点，使用原始点云计算长度")
            filtered_points = points
        
        logger.info(f"筛选后剩余{len(filtered_points)}个点")
        
        # 步骤5: 计算直线范围内点云数据的包围盒长度
        x_min_filtered = np.min(filtered_points[:, 0])
        x_max_filtered = np.max(filtered_points[:, 0])
        length = x_max_filtered - x_min_filtered
        
        logger.info(f"计算得到长度: {length:.4f}")
        
        return width, length
    
    
    def calculate_block_dimensions(self, pcd_path):
        """
        计算单元块的几何参数
        
        Args:
            pcd_path: 点云文件路径
            
        Returns:
            包含计算结果的字典 {"W0": W0, "W1": W1, "L0": L0, "L1": L1}
        """
        result = {
            "W0": None,  # 上宽
            "W1": None,  # 下宽
            "L0": None,  # 上长
            "L1": None   # 下长
        }
        
        try:
            # 1. 加载点云
            logger.info(f"加载点云文件: {pcd_path}")
            pcd = o3d.io.read_point_cloud(pcd_path)
            
            if len(pcd.points) == 0:
                raise Exception("点云为空")
            
            logger.info(f"点云加载成功，包含{len(pcd.points)}个点")
            
            
            # 3. 生成分层切片
            top_slice_z, bottom_slice_z = self._generate_slices(pcd)
            
            # 4. 提取顶部切片轮廓
            logger.info("开始提取顶部切片轮廓...")
            top_contours = []
            
            # 提取切片点，传入is_top=True
            xy_points1, slice_pcd1 = self._extract_slice_contour(pcd, top_slice_z,tolerance_percentage = self.top_slice_ratio, is_top=True)
            
            # 保存切片点云
            if len(slice_pcd1.points) > 0:
                self._save_point_cloud(slice_pcd1, f"单元块_顶部切片.ply", [0, 0, 1])
            
            if len(xy_points1) > 0:
                # 计算轮廓多边形
                contour = self._compute_contour_polygon(xy_points1)
                top_contours.append(contour)
                
                # 保存轮廓，使用顶部Z值的最大值作为反向投影的Z值
                if len(contour) > 0:
                    self._save_contour(contour, top_slice_z, f"单元块_顶部轮廓.ply", [0, 0, 1])
            
            # 5. 提取底部切片轮廓
            logger.info("开始提取底部切片轮廓...")
            bottom_contours = []
            
            # 提取切片点，传入is_top=False
            xy_points, slice_pcd = self._extract_slice_contour(pcd, bottom_slice_z, tolerance_percentage = self.bottom_slice_ratio, is_top=False)
            
            # 保存切片点云
            if len(slice_pcd.points) > 0:
                self._save_point_cloud(slice_pcd, f"单元块_底部切片.ply", [1, 0, 0])
            
            if len(xy_points) > 0:
                # 计算轮廓多边形
                contour = self._compute_contour_polygon(xy_points)
                bottom_contours.append(contour)
                
                # 保存轮廓，使用底部Z值的最小值作为反向投影的Z值
                if len(contour) > 0:
                    self._save_contour(contour, bottom_slice_z, f"单元块_底部轮廓.ply", [1, 0, 0])
            
            # 6. 计算尺寸
            logger.info("开始计算尺寸...")
            
            # 计算顶部尺寸
            if top_contours:
                # top_widths,_ = self._calculate_dimensions_from_contours(top_contours)
                # logger.info(f"拟合直线法_顶部宽度W0 (中位数): {np.median(top_widths):.6f}")
                # logger.info(f"拟合直线法_顶部长度L0 (中位数): {np.median(top_lengths):.6f}")

                top_lengths, top_widths = calculate_parallelogram_length_width(slice_pcd1, visualize=False,output_file =os.path.join(self.output_dir, '单元块_顶部边缘线.ply'),
                 y_max_distance=510,y_min_distance=410,x_max_distance=1090,x_min_distance=1030)
                
                if top_widths:
                    result["W0"] = float(np.median(top_widths))
                    logger.info(f"上宽W0 (中位数): {result['W0']:.6f}")
                
                if top_lengths:
                    result["L0"] = float(np.median(top_lengths))
                    logger.info(f"上长L0 (中位数): {result['L0']:.6f}")
            
            # 计算底部尺寸
            if bottom_contours:
                # bottom_widths,_ = self._calculate_dimensions_from_contours(slice_pcd)
                # logger.info(f"拟合直线法_底部宽度W1 (中位数): {np.median(bottom_widths):.6f}")
                # logger.info(f"拟合直线法_底部长度L1 (中位数): {np.median(bottom_lengths):.6f}")
                
                bottom_lengths, bottom_widths = calculate_parallelogram_length_width(slice_pcd, visualize=False,output_file =os.path.join(self.output_dir, '单元块_底部边缘线.ply'),
                 y_max_distance=550,y_min_distance=450,x_max_distance=1130,x_min_distance=1030)
                
                if bottom_widths:
                    result["W1"] = float(np.median(bottom_widths)+10)
                    logger.info(f"下宽W1 (中位数): {result['W1']:.6f}")
                
                if bottom_lengths:
                    result["L1"] = float(np.median(bottom_lengths)+10)
                    logger.info(f"下长L1 (中位数): {result['L1']:.6f}")
            
            # 验证结果有效性
            valid_results = [v for v in result.values() if v is not None]
            if len(valid_results) < 4:
                logger.warning(f"计算结果不完整: {result}")
            else:
                logger.info("几何参数计算成功")
            
        except Exception as e:
            logger.error(f"计算过程出错: {e}")
            raise
        
        return result

# 主函数，用于测试
if __name__ == "__main__":
    try:
        # 创建测量对象
        measurer = BlockMeasurement()
        
        measurer.alpha = 100  # Alpha形状算法的alpha参数
        
        # 测试文件路径
        test_files = [
            "2cc668be-5a40-4304-8a82-c5346f69f1e1\dyk_l.ply",
            "2cc668be-5a40-4304-8a82-c5346f69f1e1\dyk_r.ply"
        ]
        
        for pcd_path in test_files:
            if os.path.exists(pcd_path):
                logger.info(f"=== 开始处理 {pcd_path} ===")
                # 执行测量
                result = measurer.calculate_block_dimensions(pcd_path)
                
                # 打印结果
                print(f"\n=== 测量结果 ({os.path.basename(pcd_path)}) ===")
                print(f"上宽 W0: {result['W0']:.6f}")
                print(f"下宽 W1: {result['W1']:.6f}")
                print(f"上长 L0: {result['L0']:.6f}")
                print(f"下长 L1: {result['L1']:.6f}")
            else:
                logger.warning(f"文件不存在: {pcd_path}")
                
    except Exception as e:
        logger.error(f"程序执行出错: {e}")