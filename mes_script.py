import os
import re
import sys
import logging
import time
import threading
import shutil
import winreg as reg
import json
import logging.handlers
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from point_preprocess import main_point_cloud_processing
from measurement_main import RailMeasurementMain

# 配置日志
def setup_logging(log_dir='logs'):
    """设置日志记录，同时输出到控制台和文件"""
    # 确保日志目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建文件处理器，使用轮转日志文件，指定UTF-8编码以支持中文
    log_file = os.path.join(log_dir, 'point_cloud_processor.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()

def process_point_cloud_pair(file1, file2, uuid, images_dir):
    """
    处理一对点云文件
    """
    output_dir = f"results/{uuid}"
    target_json = f"device001_{uuid}.json"
    processed_files = []
    
    try:
        # 记录要处理的文件
        processed_files.append(file1)
        processed_files.append(file2)
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"创建输出目录: {output_dir}")
            except Exception as e:
                logger.error(f"创建目录失败: {e}")
                return False, []
        
        # 创建测量主类实例
        main_measurer = RailMeasurementMain(output_dir=output_dir, target_json=target_json, uuid=uuid)
        
        # 处理点云数据
        processing_result = main_point_cloud_processing(
            file1,
            file2,
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
            logger.info(f"UUID {uuid} 测量完成，结果已保存")
            return True, processed_files
        else:
            logger.error(f"UUID {uuid} 测量失败")
            return False, []
            
    except Exception as e:
        logger.error(f"处理UUID {uuid} 时发生错误: {str(e)}")
        return False, []

def match_and_process_files(images_dir, processed_files_dict):
    """
    匹配并处理符合条件的文件对
    """
    # 获取所有.txt文件
    all_files = [f for f in os.listdir(images_dir) if f.endswith('.txt')]
    
    # 按uuid分组
    uuid_groups = {}
    pattern = r'pointData-(\d+)-(\d{10})\.txt'
    
    for file in all_files:
        match = re.match(pattern, file)
        if match:
            camera_id = match.group(1)
            uuid = match.group(2)
            
            if uuid not in uuid_groups:
                uuid_groups[uuid] = {}
            
            uuid_groups[uuid][camera_id] = os.path.join(images_dir, file)
    
    # 处理每组数据
    for uuid, camera_files in uuid_groups.items():
        # 跳过已经处理或正在处理的UUID
        if uuid in processed_files_dict and processed_files_dict[uuid]['status'] in ['processing', 'processed']:
            continue
        
        logger.info(f"检查UUID: {uuid}")
        
        # 检查是否同时有两个相机的数据
        if '0' not in camera_files or '1' not in camera_files:
            logger.warning(f"UUID {uuid} 缺少相机数据，等待另一文件")
            continue
        
        # 标记为处理中
        processed_files_dict[uuid] = {'status': 'processing'}
        
        # 处理文件对
        success, processed_files = process_point_cloud_pair(
            camera_files['0'], camera_files['1'], uuid, images_dir)
        
        if success:
            processed_files_dict[uuid]['status'] = 'processed'
            processed_files_dict[uuid]['files'] = processed_files
            
            for file_path in processed_files:
                logger.info(f"关闭删除功能，不删除文件: {file_path}")
            
            # # 删除已处理的文件
            # for file_path in processed_files:
            #     try:
            #         if os.path.exists(file_path):
            #             os.remove(file_path)
            #             logger.info(f"已删除文件: {file_path}")
            #         else:
            #             logger.warning(f"文件不存在: {file_path}")
            #     except Exception as e:
            #         logger.error(f"删除文件 {file_path} 失败: {str(e)}")
        else:
            # 处理失败，移除标记
            if uuid in processed_files_dict:
                del processed_files_dict[uuid]
        
        # 添加处理间隔，避免资源占用过高
        time.sleep(10)

def process_point_cloud_files(images_dir="D:\images"):
    """
    批量处理指定路径下的点云文件
    """
    # 检查目录是否存在
    if not os.path.exists(images_dir):
        logger.error(f"目录不存在: {images_dir}")
        return
    
    # 用于记录已处理和正在处理的文件信息
    processed_files_dict = {}
    
    match_and_process_files(images_dir, processed_files_dict)
    logger.info("当前批处理完成")

class PointCloudHandler(FileSystemEventHandler):
    """
    文件系统事件处理器，用于监控目录中的新文件
    """
    def __init__(self, images_dir, processed_files_dict, processing_lock):
        self.images_dir = images_dir
        self.processed_files_dict = processed_files_dict
        self.processing_lock = processing_lock
    
    # 在on_created事件处理中增加延迟
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.txt'):
            logger.info(f"检测到新文件: {event.src_path}")
            time.sleep(2)  # 延迟2秒确保文件完全写入
            with self.processing_lock:
                match_and_process_files(self.images_dir, self.processed_files_dict)
    
    # 增加组处理后的延迟
    def match_and_process_files(images_dir, processed_files_dict):
        """
        匹配并处理符合条件的文件对
        """
        # 获取所有.txt文件
        all_files = [f for f in os.listdir(images_dir) if f.endswith('.txt')]
        
        # 按uuid分组
        uuid_groups = {}
        pattern = r'pointData-(\d+)-(\d{10})\.txt'
        
        for file in all_files:
            match = re.match(pattern, file)
            if match:
                camera_id = match.group(1)
                uuid = match.group(2)
                
                if uuid not in uuid_groups:
                    uuid_groups[uuid] = {}
                
                uuid_groups[uuid][camera_id] = os.path.join(images_dir, file)
        
        # 处理每组数据
        for uuid, camera_files in uuid_groups.items():
            # 跳过已经处理或正在处理的UUID
            if uuid in processed_files_dict and processed_files_dict[uuid]['status'] in ['processing', 'processed']:
                continue
            
            logger.info(f"检查UUID: {uuid}")
            
            # 检查是否同时有两个相机的数据
            if '0' not in camera_files or '1' not in camera_files:
                logger.warning(f"UUID {uuid} 缺少相机数据，等待另一文件")
                continue
            
            # 标记为处理中
            processed_files_dict[uuid] = {'status': 'processing'}
            
            # 处理文件对
            success, processed_files = process_point_cloud_pair(
                camera_files['0'], camera_files['1'], uuid, images_dir)
            
            if success:
                # 更新处理状态
                processed_files_dict[uuid]['status'] = 'processed'
                processed_files_dict[uuid]['files'] = processed_files
                
                # 删除已处理的文件
                for file_path in processed_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"已删除文件: {file_path}")
                        else:
                            logger.warning(f"文件不存在: {file_path}")
                    except Exception as e:
                        logger.error(f"删除文件 {file_path} 失败: {str(e)}")
            else:
                # 处理失败，移除标记
                if uuid in processed_files_dict:
                    del processed_files_dict[uuid]
            
            # 添加处理间隔，避免资源占用过高
            time.sleep(10)  # 将延迟从1秒增加到10秒

class FileMonitor:
    """
    文件监控类，持续监控指定目录
    """
    def __init__(self, images_dir="D:\images"):
        self.images_dir = images_dir
        self.processed_files_dict = {}
        self.processing_lock = threading.Lock()
        self.observer = None
        self.stop_event = threading.Event()
    
    def start_monitoring(self):
        """
        启动文件监控
        """
        # 确保监控目录存在
        if not os.path.exists(self.images_dir):
            logger.error(f"监控目录不存在: {self.images_dir}")
            try:
                os.makedirs(self.images_dir)
                logger.info(f"已创建监控目录: {self.images_dir}")
            except Exception as e:
                logger.error(f"创建监控目录失败: {e}")
                return False
        
        # 创建事件处理器
        event_handler = PointCloudHandler(
            self.images_dir, 
            self.processed_files_dict, 
            self.processing_lock
        )
        
        # 创建观察者
        self.observer = Observer()
        self.observer.schedule(event_handler, self.images_dir, recursive=False)
        
        # 启动观察者
        self.observer.start()
        logger.info(f"开始监控目录: {self.images_dir}")
        
        # 先处理目录中已有的文件
        with self.processing_lock:
            match_and_process_files(self.images_dir, self.processed_files_dict)
        
        return True
    
    def stop_monitoring(self):
        """
        停止文件监控
        """
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("文件监控已停止")
        self.stop_event.set()

def set_windows_autostart(enable=True):
    """
    设置Windows开机自启动
    """
    try:
        # 获取脚本的完整路径
        script_path = os.path.abspath(sys.argv[0])
        
        # 注册表路径
        key_path = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Run'
        key = reg.OpenKey(reg.HKEY_CURRENT_USER, key_path, 0, reg.KEY_SET_VALUE)
        
        if enable:
            # 设置自启动项
            reg.SetValueEx(key, "PointCloudProcessor", 0, reg.REG_SZ, 
                         f'"{sys.executable}" "{script_path}" --background')
            logger.info(f"已设置开机自启动: {script_path}")
        else:
            # 删除自启动项
            try:
                reg.DeleteValue(key, "PointCloudProcessor")
                logger.info("已取消开机自启动")
            except WindowsError:
                logger.info("自启动项不存在")
        
        reg.CloseKey(key)
        return True
    except Exception as e:
        logger.error(f"设置开机自启动失败: {e}")
        return False

def run_in_background():
    """
    以后台服务方式运行
    """
    # 创建守护线程运行监控
    monitor = FileMonitor(images_dir="D:/imagesbackup")
    
    # 启动监控
    if monitor.start_monitoring():
        try:
            logger.info("程序已在后台启动，按Ctrl+C退出")
            # 主循环保持程序运行
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("收到停止信号，正在停止...")
            monitor.stop_monitoring()
            logger.info("程序已停止")
        except Exception as e:
            logger.error(f"后台运行时发生错误: {e}")
            monitor.stop_monitoring()

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = sys.argv[1:]
    
    # 检查是否需要设置自启动
    if '--autostart' in args:
        set_windows_autostart(True)
        return
    
    # 检查是否需要取消自启动
    if '--no-autostart' in args:
        set_windows_autostart(False)
        return
    
    # 检查是否需要后台运行
    if '--background' in args:
        run_in_background()
        return
    
    # 默认模式：单次运行处理
    try:
        run_in_background()
        # process_point_cloud_files(images_dir="C:\images")
        # logger.info("所有点云数据处理完成")
    except KeyboardInterrupt:
        logger.info("用户中断处理")
        sys.exit(1)
    except Exception as e:
        logger.error(f"处理过程中发生严重错误: {str(e)}")
        sys.exit(1)
    input("Press Enter to continue...")
    

if __name__ == "__main__":
    main()