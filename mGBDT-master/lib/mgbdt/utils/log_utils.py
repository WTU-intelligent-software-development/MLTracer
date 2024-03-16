import logging
import os

import colorlog


class logger:
    def __init__(self, output_dir: str, name=None, loggerLevel=logging.DEBUG, consoleLevel=logging.DEBUG, dirLevel=logging.INFO):
        # 创建 logger 实例
        self.mylogger = self.getMyLogger(output_dir, name, loggerLevel, consoleLevel, dirLevel)
        self.output_dir = output_dir

    def getMyLogger(self, output_dir, name=None, loggerLevel=logging.DEBUG, consoleLevel=logging.DEBUG, dirLevel=logging.INFO):
        if output_dir:
            mylogger = logging.getLogger(name)
            mylogger.setLevel(loggerLevel)
            # 创建控制台日志 handler
            console_handler = self.set_console_handler(consoleLevel)
            file_handler = self.set_file_handler(output_dir, dirLevel)
            # 将控制台handler和文件handler添加到mylogger
            mylogger.addHandler(console_handler)
            mylogger.addHandler(file_handler)
            return mylogger
        else:
            raise ValueError("Please set log name.")

    def __del__(self):
        for hdlr in self.mylogger.handlers:
            self.mylogger.removeHandler(hdlr)

    def set_console_handler(self, consoleLevel=logging.DEBUG):
        """
        创建控制台日志 handler
        """
        console_handler = logging.StreamHandler()
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s[ %(asctime)s][%(module)s.%(funcName)s] %(message)s",
            log_colors={
                'DEBUG': 'blue',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            reset=True,
            style='%'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(consoleLevel)
        return console_handler

    def set_file_handler(self, logging_dir, dirLevel=logging.INFO):
        folder_path = os.path.dirname(logging_dir)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        fh = logging.FileHandler(logging_dir, mode="w+")
        fh.setFormatter(logging.Formatter("[ %(asctime)s][%(module)s.%(funcName)s] %(message)s"))
        fh.setLevel(dirLevel)
        return fh

    def debug(self, msg):
        self.mylogger.debug(msg)

    def info(self, msg):
        self.mylogger.info(msg)

    def warning(self, msg):
        self.mylogger.warning(msg)

    def error(self, msg):
        self.mylogger.error(msg)

    def critical(self, msg):
        self.mylogger.critical(msg)


if __name__ == "__main__":
    pass
