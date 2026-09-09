# common_transfer_functions.py
import socket
import struct
import numpy as np
import io
import os
import time
# from common_config import BUFFER_SIZE  # 从配置文件导入
from .common_config import BUFFER_SIZE  #改为导包的逻辑

# --- 辅助函数：发送NPZ文件 ---
def send_npz_file(sock: socket.socket, file_path: str):
    """
    通过给定套接字发送NPZ文件。
    协议: [文件大小 (4字节无符号整数)] + [npz文件内容]
    """
    try:
        # 1. 读取NPZ文件内容
        with open(file_path, 'rb') as f:
            npz_data = f.read()

        # 2. 发送文件大小 (4字节，无符号整数)
        file_size = len(npz_data)
        sock.sendall(struct.pack('!I', file_size))
        print(f"[发送] 文件 '{os.path.basename(file_path)}' 大小: {file_size} 字节...")

        # 3. 发送文件内容
        sock.sendall(npz_data)
        print(f"[发送] 文件 '{os.path.basename(file_path)}' 发送完成。")
        return True
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return False
    except socket.error as e:
        print(f"发送文件时发生套接字错误: {e}")
        return False
    except Exception as e:
        print(f"发送文件时发生未知错误: {e}")
        return False


# --- 辅助函数：接收NPZ文件 ---
def receive_npz_file(sock: socket.socket, save_dir: str = '.', prefix: str = 'received_'):
    """
    通过给定套接字接收NPZ文件。
    协议: [文件大小 (4字节无符号整数)] + [npz文件内容]
    """
    try:
        # 1. 接收文件大小 (4字节)
        size_bytes = b''
        while len(size_bytes) < 4:
            packet = sock.recv(4 - len(size_bytes))
            if not packet:
                return None
            size_bytes += packet

        file_size = struct.unpack('!I', size_bytes)[0]
        print(f"[接收] 准备接收文件，预计大小: {file_size} 字节...")

        # 2. 接收文件内容
        received_data = b''
        while len(received_data) < file_size:
            remaining_bytes = file_size - len(received_data)
            packet = sock.recv(min(BUFFER_SIZE, remaining_bytes))
            if not packet:
                print("[接收] 连接已关闭或无数据，文件未完整接收。")
                return None
            received_data += packet

        # 将接收到的字节数据保存为npz文件
        timestamp = int(time.time())
        output_filename = os.path.join(save_dir, f"{prefix}{timestamp}.npz")

        os.makedirs(save_dir, exist_ok=True)

        with open(output_filename, 'wb') as f:
            f.write(received_data)


        # 尝试加载并打印内容，以验证文件
        try:
            with io.BytesIO(received_data) as f:
                loaded_data = np.load(f)
                loaded_data.close()
        except Exception as e:
            print(f"[接收] 警告: 无法加载接收到的NPZ文件 '{output_filename}'，可能已损坏: {e}")

        return output_filename
    except socket.error as e:
        print(f"[接收] 接收文件时发生套接字错误: {e}")
        return None
    except Exception as e:
        print(f"[接收] 接收文件时发生未知错误: {e}")
        return None

