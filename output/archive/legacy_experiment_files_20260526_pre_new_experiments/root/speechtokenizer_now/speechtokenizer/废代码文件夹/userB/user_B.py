# user_B.py
import time
import queue

from .common_config import PORT_A_SERVER, PORT_B_SERVER
from secure_link import SecureSender, SecureReceiver


def start_user_b_secure_link():
    """
    启动 User B 的 SecureLink（纯内存队列）

    返回：
        send_q: B -> A 的发送队列
        recv_q: A -> B 的接收队列
    """
    send_q = queue.Queue(maxsize=64)
    recv_q = queue.Queue(maxsize=64)

    # B 接收来自 A 的数据
    rx = SecureReceiver(
        recv_q=recv_q,
        listen_port=PORT_B_SERVER,
        tmp_dir="B_tmp_recv"
    )

    # B 向 A 发送数据
    tx = SecureSender(
        send_q=send_q,
        peer_port=PORT_A_SERVER,
        tmp_dir="B_tmp_send"
    )

    rx.start()
    tx.start()

    return send_q, recv_q


if __name__ == "__main__":
    print("[UserB] secure link running")

    start_user_b_secure_link()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[UserB] exit")
