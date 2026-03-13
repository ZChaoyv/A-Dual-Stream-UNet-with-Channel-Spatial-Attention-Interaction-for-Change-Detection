import sys
import time


################################################################################
#                                                                              
#              📜 SYSTEM LOGGING & EXECUTION TIMER TOOLS                       
#                  📜 系统日志记录与执行计时工具中心                               
#                                                                              
#   Description: This script manages real-time console logging to files and     
#   provides a high-precision timer for tracking training progress/ETAs.        
#   代码说明：该脚本负责将终端输出实时记录至文件，并提供高精度计时器以追踪训练进度。       
#                                                                              
################################################################################


# ==============================================================================
# [Logger] Dual-output logging (Console + File)
# [Logger] 双向输出日志（终端 + 文件）
# ==============================================================================
class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        # Initialize log file with timestamp / 使用时间戳初始化日志文件
        self.write('================ (%s) ================\n' % now)

    def write(self, message):
        """Simultaneously print to terminal and append to file / 同时输出至终端并追加至文件"""
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    def write_dict(self, dict):
        """Format and log numeric dictionaries (e.g., loss values) / 格式化并记录数值字典（如 Loss）"""
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        """Log configuration dictionaries as strings / 以字符串形式记录配置参数字典"""
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()


# ==============================================================================
# [Timer] Performance tracking and ETA estimation
# [Timer] 性能追踪与预计剩余时间 (ETA) 估算
# ==============================================================================
class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        """
        Update ETA based on current completion percentage
        根据当前完成百分比更新预计剩余时间
        """
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)

    def str_estimated_complete(self):
        """Return human-readable finish time / 返回易读的预计完成时间"""
        return str(time.ctime(self.est_finish))

    def str_estimated_remaining(self):
        """Return remaining hours / 返回剩余小时数"""
        return str(self.est_remaining/3600) + 'h'

    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        """Get time elapsed for current stage / 获取当前阶段已耗时"""
        return time.time() - self.stage_start

    def reset_stage(self):
        """Restart stage timer / 重置阶段计时"""
        self.stage_start = time.time()

    def lapse(self):
        """Get elapsed time and reset stage / 获取耗时并立即重置阶段计时"""
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out