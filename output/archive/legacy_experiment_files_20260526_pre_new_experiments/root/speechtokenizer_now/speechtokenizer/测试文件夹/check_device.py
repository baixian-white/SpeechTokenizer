import sounddevice as sd

print(f"{'ID':<4} {'类型':<6} {'协议':<15} {'名称'}")
print("-" * 70)

devices = sd.query_devices()

for i, dev in enumerate(devices):
    # 修正点：使用 query_hostapis 而不是 query_host_apis
    host_api = sd.query_hostapis()[dev['hostapi']]['name']
    
    # 过滤掉非默认的复杂接口，只看常用的
    if dev['max_input_channels'] > 0:
        print(f"{i:<4} [麦克风] {host_api:<15} {dev['name']}")
    elif dev['max_output_channels'] > 0:
        print(f"{i:<4} [扬声器] {host_api:<15} {dev['name']}")