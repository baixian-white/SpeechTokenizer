# encryption_utils.py
import numpy as np
import os

def encrypt_npz_first_array_only(input_npz_path, key_txt_path, output_npz_path):
    """
    加密函数：仅加密 npz 文件中的第一个数组
    """
    print(f" 开始处理: {input_npz_path}")
    data = np.load(input_npz_path)
    output_data = {}

    # 获取第一个数组的名称
    if not data.files:
        raise ValueError("NPZ 文件中没有任何数组")
    target_array_name = data.files[0]

    # 读取密钥流
    try:
        with open(key_txt_path, 'r') as f:
            full_key_str = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"密钥文件 '{key_txt_path}' 不存在。")

    # 遍历所有数组
    for array_name in data.files:
        original_arr = data[array_name]

        # --- 核心修改：判断是否为第一个数组 ---
        if array_name != target_array_name:
            # 如果不是第一个数组，直接原样保存，不处理
            output_data[array_name] = original_arr
            continue
        # ----------------------------------

        # 下面是对目标数组（第一个数组）的加密逻辑

        # 检查是否为整数类型
        if not np.issubdtype(original_arr.dtype, np.integer):
            print(f" 警告: 第一个数组 '{array_name}' 不是整数类型，跳过处理。")
            output_data[array_name] = original_arr
            continue

        # 数据类型转换处理 (int -> int64 -> uint64 view)
        if original_arr.dtype != np.int64:
            original_arr = original_arr.astype(np.int64)

        # 使用 uint64 视图进行位运算，防止 "int too big" 错误
        arr_view_as_uint64 = original_arr.view(np.uint64)

        total_elements = arr_view_as_uint64.size
        total_bits_needed = total_elements * 64

        # 截取密钥片段（只针对这就这一个数组，所以从头取即可）
        key_segment = full_key_str[:total_bits_needed]

        if len(key_segment) < total_bits_needed:
            raise ValueError(
                f"密钥文件长度不足。数组 '{array_name}' 需要 {total_bits_needed} 位，"
                f"但密钥文件仅有 {len(full_key_str)} 位。"
            )

        # 将 01 字符串转为 uint64 掩码数组
        key_values = []
        for i in range(0, len(key_segment), 64):
            chunk = key_segment[i:i + 64]
            val = int(chunk, 2)
            key_values.append(val)

        key_arr = np.array(key_values, dtype=np.uint64)
        key_arr = key_arr.reshape(arr_view_as_uint64.shape)

        # 执行异或
        encrypted_arr_uint64 = np.bitwise_xor(arr_view_as_uint64, key_arr)

        # 转回 int64 视图
        final_encrypted_arr = encrypted_arr_uint64.view(np.int64)

        output_data[array_name] = final_encrypted_arr

    # 保存
    np.savez_compressed(output_npz_path, **output_data)



def decrypt_npz_first_array_only(encrypted_npz_path, key_txt_path, output_npz_path):
    """
    解密函数：逻辑与加密完全一致
    """
    encrypt_npz_first_array_only(encrypted_npz_path, key_txt_path, output_npz_path)



def xor_encrypt_int_ndarray(arr: np.ndarray, key_txt_path: str) -> np.ndarray:
    """
    对整数 ndarray 进行 XOR 加密（与你原 npz 版本逻辑完全一致）
    返回：加密后的 ndarray（int64）
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise TypeError("只支持整数 ndarray 加密")

    # 转成 int64，保证 64-bit 对齐
    if arr.dtype != np.int64:
        arr = arr.astype(np.int64)

    # 读取密钥流
    try:
        with open(key_txt_path, 'r') as f:
            full_key_str = f.read().strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"密钥文件 '{key_txt_path}' 不存在。")

    # uint64 视图
    arr_u64 = arr.view(np.uint64)

    total_elements = arr_u64.size
    total_bits_needed = total_elements * 64

    if len(full_key_str) < total_bits_needed:
        raise ValueError(
            f"密钥长度不足，需要 {total_bits_needed} bits，"
            f"但只有 {len(full_key_str)} bits"
        )

    # 构造 key 掩码
    key_values = []
    for i in range(0, total_bits_needed, 64):
        chunk = full_key_str[i:i + 64]
        key_values.append(int(chunk, 2))

    key_arr = np.array(key_values, dtype=np.uint64).reshape(arr_u64.shape)

    # XOR
    encrypted_u64 = np.bitwise_xor(arr_u64, key_arr)

    # 转回 int64
    return encrypted_u64.view(np.int64)


def xor_decrypt_int_ndarray(arr: np.ndarray, key_txt_path: str) -> np.ndarray:
    """
    XOR 解密：与加密完全相同
    """
    return xor_encrypt_int_ndarray(arr, key_txt_path)
