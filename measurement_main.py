from audioop import avg
import open3d as o3d
import numpy as np
import os
import json
import logging
import sys
import time
from point_segmentation import PointCloudSegmenter
from measurement_1 import RailMeasurement
from measurement_2 import RailInclinationMeasurement
from measurement_3 import BlockMeasurement
from point_preprocess import main_point_cloud_processing,extract_largest_cluster

# 创建一个自定义的StreamHandler，用于捕获print输出到日志
class PrintToLoggerHandler(logging.Handler):
    def __init__(self, logger, level=logging.INFO):
        super().__init__(level)
        self.logger = logger
        self.original_stdout = None
    
    def emit(self, record):
        msg = self.format(record)
        self.logger.log(record.levelno, msg)
    
    def start(self):
        """开始捕获print输出"""
        self.original_stdout = sys.stdout
        sys.stdout = self
    
    def stop(self):
        """停止捕获print输出"""
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
    
    def write(self, message):
        """重写write方法，将print输出写入日志"""
        if message.strip():
            self.logger.info(message.strip())
    
    def flush(self):
        """重写flush方法"""
        pass

# 配置日志
def configure_logging(log_file=None, log_level=logging.INFO):
    # 基础配置
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 获取根logger
    root_logger = logging.getLogger()
    
    # 清除已有的处理器，避免重复输出
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志文件目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        print(f"日志文件已配置: {log_file}")
    
    return logging.getLogger('rail_measurement_main')

# 默认配置logger，稍后在类初始化时会重新配置
logger = configure_logging()

class RailMeasurementMain:
    """轨枕测量主函数类"""
    
    def __init__(self, output_dir="measurements_results", target_json="measurements.json", uuid="", log_file=None, log_level=logging.INFO):
        """
        初始化测量主类
        
        Args:
            output_dir (str): 输出结果目录
            log_file (str): 日志文件路径，None表示不输出到文件
            log_level (int): 日志级别
        """
        self.output_dir = output_dir
        self.uuid = uuid
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        self.target_json = target_json
        
        # 配置日志
        if log_file:
            # 如果用户指定了日志文件路径，直接使用
            self.log_file = log_file
        else:
            # 否则默认在输出目录下创建日志文件
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            self.log_file = os.path.join(output_dir, f"measurement_log_{timestamp}.log")
        
        # 重新配置logger
        global logger
        logger = configure_logging(self.log_file, log_level)
        
        # 创建print重定向处理器
        self.print_handler = PrintToLoggerHandler(logger)
        # 开始捕获print输出
        self.print_handler.start()
        
        logger.info(f"创建输出目录: {output_dir}")
        logger.info(f"日志文件路径: {self.log_file}")
        
        # 初始化各个测量模块
        self.segmenter = PointCloudSegmenter(output_dir)
        self.rail_measurer = RailMeasurement(output_dir)
        self.inclination_measurer = RailInclinationMeasurement(output_dir)
        self.block_measurer = BlockMeasurement(output_dir)
    
    def __del__(self):
        """析构函数，确保在对象销毁时停止捕获print输出"""
        try:
            if hasattr(self, 'print_handler'):
                self.print_handler.stop()
        except:
            pass  # 忽略析构函数中的异常
    
    def load_point_cloud(self, file_path):
        """
        加载点云数据，支持PLY和TXT文件格式
        
        Args:
            file_path (str): 点云文件路径
            
        Returns:
            o3d.geometry.PointCloud: 加载的点云数据
        """
        try:
            # 首先检查文件是否存在
            if not os.path.exists(file_path):
                # 如果文件不存在，尝试在data目录中查找
                data_path = os.path.join("data", os.path.basename(file_path))
                if os.path.exists(data_path):
                    logger.info(f"在data目录中找到文件: {data_path}")
                    file_path = data_path
                else:
                    logger.error(f"点云文件不存在: {file_path} 和 {data_path}")
                    return None
            
            # 检查文件扩展名，区分处理PLY和TXT文件
            _, file_ext = os.path.splitext(file_path)
            
            if file_ext.lower() == '.ply':
                # PLY文件直接使用Open3D读取
                pcd = o3d.io.read_point_cloud(file_path)
                logger.info(f"成功加载PLY点云: {file_path}, 点数: {len(pcd.points)}")
                return pcd
            elif file_ext.lower() == '.txt':
                # TXT文件需要处理编码问题和数据格式
                try:
                    logger.info(f"检测到TXT文件，尝试使用UTF-8编码读取: {file_path}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    logger.info(f"成功使用UTF-8读取TXT文件，共 {len(lines)} 行")
                except UnicodeDecodeError:
                    logger.info(f"UTF-8编码读取失败，尝试使用GBK编码: {file_path}")
                    with open(file_path, 'r', encoding='gbk') as f:
                        lines = f.readlines()
                    logger.info(f"成功使用GBK读取TXT文件，共 {len(lines)} 行")
            
                # 解析TXT文件数据
                points = []
                skipped_lines = 0  # 计数器：记录跳过的行数
                
                for line in lines:
                    # 对于逗号分隔的CSV格式数据，使用逗号作为分隔符
                    parts = line.strip().split(',')
                    
                    # 确保解析出至少4个字段（x, y, z, brightness）
                    if len(parts) >= 4:  # 确保至少有四个数据（x,y,z,brightness）
                        try:
                            # 处理最后一个字段可能包含换行符的情况
                            # 提取前三个坐标值，忽略亮度值
                            # 去除任何可能的非数字字符，如换行符
                            x_str = parts[0].strip()
                            y_str = parts[1].strip()
                            z_str = parts[2].strip()
                            
                            # 转换为浮点数
                            x, y, z = float(x_str), float(y_str), float(z_str)
                            
                            # 检查是否包含NaN值，如果有则跳过
                            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                                points.append([x, y, z])
                            else:
                                skipped_lines += 1
                        except ValueError:
                            # 如果无法转换为数字，跳过该行
                            skipped_lines += 1
                    else:
                        # 行数不足四个数据，计入跳过行数
                        skipped_lines += 1
                
                logger.info(f"解析完成，共跳过 {skipped_lines} 行无效数据，有效点数: {len(points)}")
                
                # 创建Open3D点云对象
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                logger.info(f"成功创建TXT点云对象，点数: {len(pcd.points)}")
                return pcd
            else:
                logger.error(f"不支持的文件格式: {file_ext}")
                return None
        except Exception as e:
            logger.error(f"加载点云失败: {e}")
            return None
    
    def step1_point_cloud_segmentation(self, pcd):
        """
        第一步：点云分割与预处理
        
        Args:
            pcd (o3d.geometry.PointCloud): 输入点云
            
        Returns:
            dict: 包含分割后的子点云
        """
        logger.info("开始点云分割与预处理")
        
        try:
            # 执行分割
            segments = self.segmenter.segment(pcd, align=False)
            
            # 提取所需子点云
            required_segments = {
                'cgt_l': segments.get('cgt_l', o3d.geometry.PointCloud()),
                'cgt_r': segments.get('cgt_r', o3d.geometry.PointCloud()),
                'cgt_m': segments.get('cgt_m', o3d.geometry.PointCloud()),
                'dyk_l': segments.get('dyk_l', o3d.geometry.PointCloud()),
                'dyk_r': segments.get('dyk_r', o3d.geometry.PointCloud())
            }
            
            # 记录分割结果统计
            for name, segment in required_segments.items():
                logger.info(f"分割结果 {name}: {len(segment.points)} 点")
            
            return required_segments
        except Exception as e:
            logger.error(f"点云分割失败: {e}")
            return None
    
    def step2_critical_parameters_measurement(self, segments):
        """
        第二步：关键参数测量
        
        Args:
            segments (dict): 分割后的子点云
            
        Returns:
            dict: 包含左右侧测量结果
        """
        logger.info("开始关键参数测量")
        
        results = {'left': {}, 'right': {}}
        
        try:
            # 1. 左侧测量
            logger.info("开始左侧参数测量")
            # 为左侧测量创建单独的输出子目录
            left_output_dir = os.path.join(self.output_dir, "left_measurements")
            os.makedirs(left_output_dir, exist_ok=True)
            self.rail_measurer.output_dir = left_output_dir
            self.inclination_measurer.output_dir = left_output_dir
            self.block_measurer.output_dir = left_output_dir
            
            # 1.1 对称度、高度、侧面宽度测量
            left_component_results = self.rail_measurer.measure_rail_components(
                segments['cgt_l'], segments['dyk_l'], test_mode=True, is_right_side=False
            )
            
            if left_component_results:
                results['left']['s1s2'] = abs(left_component_results['symmetry_s1']['value'])
                results['left']['h'] = abs(left_component_results['height_h']['value'])
                results['left']['w'] = abs(left_component_results['side_width_we']['value'])
            
            # 1.2 倾斜角测量
            left_angle_result = self.inclination_measurer.calculate_inclination_angle(
                segments['cgt_l'], test_mode=True
            )
            
            if left_angle_result['success']:
                results['left']['angle'] = left_angle_result['value']
            
            # 1.3 复合单元块尺寸测量
            # 先保存分割后的点云为临时文件
            left_dyk_path = os.path.join(left_output_dir, "temp_dyk_l.ply")
            o3d.io.write_point_cloud(left_dyk_path, segments['dyk_l'])
            left_block_results = self.block_measurer.calculate_block_dimensions(
                left_dyk_path
            )
            
            if left_block_results:
                results['left']['w0'] = left_block_results.get('W0', 0)
                results['left']['w1'] = left_block_results.get('W1', 0)
                results['left']['l0'] = left_block_results.get('L0', 0)
                results['left']['l1'] = left_block_results.get('L1', 0)
            
            # 2. 右侧测量
            logger.info("开始右侧参数测量")
            # 为右侧测量创建单独的输出子目录
            right_output_dir = os.path.join(self.output_dir, "right_measurements")
            os.makedirs(right_output_dir, exist_ok=True)
            self.rail_measurer.output_dir = right_output_dir
            self.inclination_measurer.output_dir = right_output_dir
            self.block_measurer.output_dir = right_output_dir
            
            # 2.1 对称度、高度、侧面宽度测量
            right_component_results = self.rail_measurer.measure_rail_components(
                segments['cgt_r'], segments['dyk_r'], test_mode=True, is_right_side=True
            )
            
            if right_component_results:
                results['right']['s1s2'] = abs(right_component_results['symmetry_s1']['value'])
                results['right']['h'] = abs(right_component_results['height_h']['value'])
                results['right']['w'] = abs(right_component_results['side_width_we']['value'])
            
            # 2.2 倾斜角测量
            right_angle_result = self.inclination_measurer.calculate_inclination_angle(
                segments['cgt_r'], test_mode=True
            )
            
            if right_angle_result['success']:
                results['right']['angle'] = right_angle_result['value']
            
            # 2.3 复合单元块尺寸测量
            right_dyk_path = os.path.join(right_output_dir, "temp_dyk_r.ply")
            o3d.io.write_point_cloud(right_dyk_path, segments['dyk_r'])
            right_block_results = self.block_measurer.calculate_block_dimensions(
                right_dyk_path
            )
            
            if right_block_results:
                results['right']['w0'] = right_block_results.get('W0', 0)
                results['right']['w1'] = right_block_results.get('W1', 0)
                results['right']['l0'] = right_block_results.get('L0', 0)
                results['right']['l1'] = right_block_results.get('L1', 0)
            
            logger.info("关键参数测量完成")
            return results
        except Exception as e:
            logger.error(f"参数测量失败: {e}")
            return None
    
    def step3_result_processing(self, measurements):
        """
        第三步：结果处理与输出
        
        Args:
            measurements (dict): 测量结果
            
        Returns:
            bool: 处理是否成功
        """
        logger.info("开始结果处理与输出")
        
        try:
            # 1. 保存左右侧结果
            # 1. 保存左右侧结果
            left_output_dir = os.path.join(self.output_dir, "left_measurements")
            os.makedirs(left_output_dir, exist_ok=True)
            left_result_path = os.path.join(left_output_dir, "left_measurement_results.json")

            right_output_dir = os.path.join(self.output_dir, "right_measurements")
            os.makedirs(right_output_dir, exist_ok=True)
            right_result_path = os.path.join(right_output_dir, "right_measurement_results.json")

            average_output_dir = os.path.join(self.output_dir, "average_measurements")
            os.makedirs(average_output_dir, exist_ok=True)
            
            # 格式化左右侧结果
            left_result = {
                "sn": "device001",
                "uuid": self.uuid,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "w0": measurements['left'].get('w0', 0.0),
                "w1": measurements['left'].get('w1', 0.0),
                "l0": measurements['left'].get('l0', 0.0),
                "l1": measurements['left'].get('l1', 0.0),
                "h": measurements['left'].get('h', 0.0),
                "s1s2": measurements['left'].get('s1s2', 0.0),
                "w": measurements['left'].get('w', 0.0),
                "angle": measurements['left'].get('angle', 0.0)
            }
            
            right_result = {
                "sn": "device001",
                "uuid": self.uuid,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "w0": measurements['right'].get('w0', 0.0),
                "w1": measurements['right'].get('w1', 0.0),
                "l0": measurements['right'].get('l0', 0.0),
                "l1": measurements['right'].get('l1', 0.0),
                "h": measurements['right'].get('h', 0.0),
                "s1s2": measurements['right'].get('s1s2', 0.0),
                "w": measurements['right'].get('w', 0.0),
                "angle": measurements['right'].get('angle', 0.0)
            }
            
            # 保存JSON文件
            with open(left_result_path, 'w', encoding='utf-8') as f:
                json.dump(left_result, f, indent=2, ensure_ascii=False)
            logger.info(f"左侧测量结果已保存至: {left_result_path}")
            
            with open(right_result_path, 'w', encoding='utf-8') as f:
                json.dump(right_result, f, indent=2, ensure_ascii=False)
            logger.info(f"右侧测量结果已保存至: {right_result_path}")
            
            # 2. 计算平均值并生成平均值结果
            avg_result = {
                # 直接保留不能平均的参数
                "sn": "device001",
                "uuid": self.uuid,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
            # 只对数值型参数计算平均值或选择较大值
            numeric_keys = ["w0", "w1", "l0", "l1", "h", "w", "angle"]
            # 定义差距阈值（当两个值的相对差距超过20%时认为差距较大）
            threshold_ratio = 0.2
            
            for key in numeric_keys:
                # 获取左右值并处理None的情况
                left_value = left_result.get(key, 0.0)
                right_value = right_result.get(key, 0.0)
                
                # 处理None值，将其转换为0.0
                if left_value is None:
                    left_value = 0.0
                    logger.warning(f"参数 {key} 左侧值为None，已转换为0.0")
                if right_value is None:
                    right_value = 0.0
                    logger.warning(f"参数 {key} 右侧值为None，已转换为0.0")
                
                # 判断两个值之间的差距是否较大
                # 避免除以零的情况
                if left_value == 0 and right_value == 0:
                    # 两个值都是0，结果也为0
                    avg_result[key] = 0.0
                else:
                    # 计算相对差距
                    max_val = max(left_value, right_value)
                    min_val = min(left_value, right_value)
                    # 使用非零值作为基准计算相对差距
                    base_val = max_val if max_val != 0 else min_val
                    relative_diff = abs(left_value - right_value) / base_val
                    
                    if relative_diff > threshold_ratio:
                        # 差距较大，选择较大的值
                        avg_result[key] = max_val
                        logger.info(f"参数 {key} 左右值差距较大（{left_value} vs {right_value}），选择较大值: {max_val}")
                    else:
                        # 差距不大，计算平均值
                        avg_result[key] = (left_value + right_value) / 2

             # 只对数值型参数计算平均值或选择较大值
            S1S2_keys = ["s1s2"]
            # 定义差距阈值（当两个值的相对差距超过20%时认为差距较大）
            threshold_ratio = 0.3
            
            for key in S1S2_keys:
                # 获取左右值并处理None的情况
                left_value = left_result.get(key, 0.0)
                right_value = right_result.get(key, 0.0)
                
                # 处理None值，将其转换为0.0
                if left_value is None:
                    left_value = 0.0
                    logger.warning(f"参数 {key} 左侧值为None，已转换为0.0")
                if right_value is None:
                    right_value = 0.0
                    logger.warning(f"参数 {key} 右侧值为None，已转换为0.0")
                
                # 判断两个值之间的差距是否较大
                # 避免除以零的情况
                if left_value == 0 and right_value == 0:
                    # 两个值都是0，结果也为0
                    avg_result[key] = 0.0
                else:
                    # 计算相对差距，使用绝对值比较大小
                    abs_left = abs(left_value)
                    abs_right = abs(right_value)
                    max_abs_val = max(abs_left, abs_right)
                    min_abs_val = min(abs_left, abs_right)
                    # 使用非零绝对值作为基准计算相对差距
                    base_val = max_abs_val if max_abs_val != 0 else min_abs_val
                    relative_diff = abs(abs_left - abs_right) / base_val
                    
                    if relative_diff > threshold_ratio:
                        # 差距较大，选择绝对值较小值
                        if abs_left >= abs_right:
                            selected_val = abs_right
                        else:
                            selected_val = abs_left
    
                        avg_result[key] = selected_val
                        logger.info(f"参数 {key} 左右值差距较大（{left_value} vs {right_value}），选择绝对值较小值: {selected_val}")
                    else:
                        # 差距不大，计算平均值
                        avg_result[key] = (abs_left + abs_right) / 2
            
            # 保存平均值结果到原路径
            avg_result_path = os.path.join(average_output_dir, "average_measurement_results.json")
            # with open(self.target_json, 'w', encoding='utf-8') as f:
            #     json.dump(avg_result, f, indent=2, ensure_ascii=False)
            # logger.info(f"平均值测量结果已保存至: {self.target_json}")
            
            # 获取文件名
            file_name = os.path.basename(self.target_json)
            # 定义额外的保存路径
            inner_result_path = f"D:\\0116_test_result\\{self.uuid}\\{file_name}"
            # mes_result_path = f"D:\\measure_result\\mesResult\\{file_name}"
            
            # 确保目录d“
            
            if not os.path.exists(f"D:\\measure_result\\{self.uuid}"):
                os.makedirs(f"D:\\measure_result\\{self.uuid}")
            for path in [inner_result_path, avg_result_path]:
                dir_path = os.path.dirname(path)
                if not os.path.exists(dir_path):
                    try:
                        os.makedirs(dir_path)
                        logger.info(f"创建目录: {dir_path}")
                    except Exception as e:
                        logger.error(f"创建目录失败: {e}")
                
                # 保存文件
                try:
                    with open(path, 'w', encoding='utf-8') as f:
                        json.dump(avg_result, f, indent=2, ensure_ascii=False)
                    logger.info(f"平均值测量结果已保存至: {path}")
                except Exception as e:
                    logger.error(f"保存文件失败 {path}: {e}")
            
            # 3. 合并所有线的点云数据
            logger.info("开始合并线的点云数据")
            
            # 创建合并后的点云对象（仅用于线数据）
            merged_lines_pcd = o3d.geometry.PointCloud()
            
            # 定义要扫描的目录
            directories_to_scan = [
                left_output_dir,
                right_output_dir,
            ]
            
            # 定义线的关键词列表
            line_keywords = ['边缘线', '对称轴', '法向量',"轮廓","平面"]
            
            # 扫描并加载所有线的点云文件
            total_line_points = 0
            loaded_line_files = 0
            
            for directory in directories_to_scan:
                if not os.path.exists(directory):
                    logger.warning(f"目录不存在: {directory}")
                    continue
                
                logger.info(f"扫描目录: {directory}")
                
                # 获取目录中的所有.ply文件
                try:
                    ply_files = [f for f in os.listdir(directory) if f.endswith('.ply')]
                    
                    for ply_file in ply_files:
                        # 检查文件名是否包含线的关键词
                        is_line_file = any(keyword in ply_file for keyword in line_keywords)
                        
                        # 跳过非线文件和临时文件
                        if not is_line_file or "temp_" in ply_file:
                            logger.debug(f"跳过非线点云: {ply_file}")
                            continue
                        
                        file_path = os.path.join(directory, ply_file)
                        
                        # 加载点云文件
                        try:
                            pcd = o3d.io.read_point_cloud(file_path)
                            
                            # 检查点云是否有颜色信息
                            if len(pcd.colors) > 0:
                                # 将线的点云添加到合并点云中
                                merged_lines_pcd += pcd
                                total_line_points += len(pcd.points)
                                loaded_line_files += 1
                                logger.info(f"已加载线点云: {ply_file}, 点数: {len(pcd.points)}")
                            else:
                                logger.debug(f"跳过无线条颜色点云: {ply_file}")
                        except Exception as e:
                            logger.warning(f"加载点云文件失败: {ply_file}, 错误: {e}")
                except Exception as e:
                    logger.error(f"扫描目录失败: {directory}, 错误: {e}")
            
            # 保存合并后的线点云
            if total_line_points > 0:
                merged_pcd_path = os.path.join(average_output_dir, "特征线.ply")
                try:
                    o3d.io.write_point_cloud(merged_pcd_path, merged_lines_pcd)
                    logger.info(f"合并的线点云已保存至: {merged_pcd_path}")
                    logger.info(f"总共合并了 {loaded_line_files} 个线点云文件，总点数: {total_line_points}")
                except Exception as e:
                    logger.error(f"保存合并线点云失败: {e}")
            else:
                logger.warning("未找到线的点云文件进行合并")
            
            logger.info("结果处理与输出完成")
            return True
        except Exception as e:
            logger.error(f"结果处理失败: {e}")
            return False
    
    def run_measurement_pipeline(self, pcd):
        """
        运行完整的测量流程
        
        Args:
            pcd (o3d.geometry.PointCloud): 输入点云数据
            
        Returns:
            bool: 测量是否成功
        """
        logger.info("开始轨枕点云测量流程")
        
        try:
            # 步骤1: 加载点
            if pcd is None:
                raise ValueError("点云加载失败")
            
            # 步骤2: 点云分割与预处理
            segments = self.step1_point_cloud_segmentation(pcd)
            if segments is None:
                raise ValueError("点云分割失败")
            
            # 步骤3: 关键参数测量
            measurements = self.step2_critical_parameters_measurement(segments)
            if measurements is None:
                raise ValueError("参数测量失败")
            
            # 步骤4: 结果处理与输出
            success = self.step3_result_processing(measurements)
            
            if success:
                logger.info("轨枕点云测量流程执行成功")
            else:
                logger.error("轨枕点云测量流程执行失败")
            
            return success
        except Exception as e:
            logger.error(f"测量流程失败: {e}")
            return False

# 独立运行测试
if __name__ == "__main__":
    # 测试入口
    import sys
    
    logger.info("启动轨枕点云测量主程序")

    if len(sys.argv) > 1:
        default_file1 = sys.argv[1]
        default_file2 = sys.argv[2]
        output_dir = sys.argv[3]
        target_json = sys.argv[4]
        uuid = sys.argv[5]
        log_file = sys.argv[6] if len(sys.argv) > 6 else None
    else:
        # 默认测试文件路径
        default_file1 = r"C:/images/pointData-0-6da493e4-04c1-4724-9bbb-71c437717ef7.txt"
        default_file2 = r"C:/images/pointData-1-6da493e4-04c1-4724-9bbb-71c437717ef7.txt"
        output_dir = "results"
        target_json = "measurements.json"
        uuid = "test_uuid"
        log_file = None  # 默认在output_dir下创建带时间戳的日志文件
    
    # 创建测量主类实例
    main_measurer = RailMeasurementMain(
        output_dir=output_dir, 
        target_json=target_json, 
        uuid=uuid,
        log_file=log_file
    )
    # 确定输入点云文件路径


    processing_result = main_point_cloud_processing(
        default_file1, 
        default_file2,
        save_downsampled=True,
        save_cluster=True,
        outputdir=output_dir,
        downsampled_filename='01下采样.ply',
        cluster_filename='02聚类去噪.ply',
        uuid=uuid
    )
    
    # 运行测量流程
    success = main_measurer.run_measurement_pipeline(processing_result['largest_cluster_pcd'])
    
    if success:
        logger.info("测量完成，结果已保存")
    else:
        logger.error("测量失败")
        sys.exit(1)

# if __name__ == "__main__":
#     file_path = r"C://Users//30583//Desktop//1106上午测试数据//722dc861-e3be-4868-b22f-7f3dde7f88dd//02聚类去噪.ply"
#     pcd = o3d.io.read_point_cloud(file_path)
#     main_measurer = RailMeasurementMain(
#         output_dir="2cc668be-5a40-4304-8a82-c5346f69f1e1", 
#         target_json="measurements.json", 
#         uuid="2cc668be-5a40-4304-8a82-c5346f69f1e1",
#         log_file=None 
#     )
#     success = main_measurer.run_measurement_pipeline(pcd)
