import sounddevice as sd
import scipy.io.wavfile as wavfile
import time

# === 配置 ===
SPK_ID = 5                  # 你刚才用的扬声器 ID (MME)
WAV_FILE = "test_mic_1.wav" # 刚才录好的文件

print(f"🔊 正在尝试通过设备 ID [{SPK_ID}] 播放文件: {WAV_FILE} ...")

try:
    # 1. 读取刚才录好的文件
    samplerate, data = wavfile.read(WAV_FILE)
    print(f"   - 文件采样率: {samplerate} Hz")
    print(f"   - 数据长度: {len(data)}")

    # 2. 播放
    print("👉 请注意听耳机里有没有声音！")
    sd.play(data, samplerate, device=SPK_ID)
    
    # 3. 等待播放完成
    sd.wait()
    print("✅ 播放指令执行完毕。")

except Exception as e:
    print(f"❌ 播放报错: {e}")