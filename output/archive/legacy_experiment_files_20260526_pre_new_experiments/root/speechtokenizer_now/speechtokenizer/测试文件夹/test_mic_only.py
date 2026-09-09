import sounddevice as sd
import scipy.io.wavfile as wavfile
import numpy as np

# === 配置 ===
MIC_ID = 1          # 你刚才用的麦克风 ID
DURATION = 5        # 录音时长 5 秒
SAMPLE_RATE = 16000 # 采样率

print(f"🎤 正在尝试从设备 ID [{MIC_ID}] 录音 {DURATION} 秒...")
print("请对着麦克风大声说话！(1...2...3...4...5)")

# 开始录音 (单声道)
try:
    recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                       samplerate=SAMPLE_RATE, 
                       channels=1, 
                       device=MIC_ID, 
                       dtype='float32')
    sd.wait()  # 等待录音结束
    
    # 保存文件
    filename = f"test_mic_{MIC_ID}.wav"
    wavfile.write(filename, SAMPLE_RATE, recording)
    print(f"✅ 录音结束！已保存为: {filename}")
    print(f"👉 请立即去文件夹里打开 {filename} 听一下！")
    
    # 简单的音量检测
    volume = np.max(np.abs(recording))
    print(f"📊 检测到的最大音量值: {volume:.4f}")
    if volume < 0.01:
        print("❌ 警告：音量极低，看起来像是静音！说明 ID 选错了！")
    else:
        print("✅ 音量正常，麦克风工作良好。")

except Exception as e:
    print(f"❌ 录音报错: {e}")