# common_config.py
HOST = '127.0.0.1'  # 或者使用你的局域网IP地址
PORT_A_SERVER = 12345 # 用户A的服务器监听端口，用户B的客户端连接此端口
PORT_B_SERVER = 12346 # 用户B的服务器监听端口，用户A的客户端连接此端口
BUFFER_SIZE = 4096  # 接收缓冲区大小


#秘钥位置的代码需要改为userA/46.txt
import os
BASE_DIR = os.path.dirname(__file__)
KEY_FILENAME = os.path.join(BASE_DIR, "46.txt")
