# -*- coding:utf-8 -*-
import logging
import datetime
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, "../")))

# 第一步，创建一个logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) #INFO/DEBUG

now_date = datetime.datetime.now()
now_date = now_date.strftime('%Y-%m-%d_%H-%M-%S')
# 第二步，创建一个handler，用于写入日志文件

if not os.path.isdir("./log"):
    os.mkdir("./log")

file_handler = logging.FileHandler('./log/' + str(now_date) + '.log', mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
)
# 添加handler到logger中
logger.addHandler(file_handler)

# 第三步，创建一个handler，用于输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s - %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
)
logger.addHandler(console_handler)

# logger.debug('this is a logger debug message')
# logger.info('this is a logger info message')
# logger.warning('this is a logger warning message')
# logger.error('this is a logger error message')