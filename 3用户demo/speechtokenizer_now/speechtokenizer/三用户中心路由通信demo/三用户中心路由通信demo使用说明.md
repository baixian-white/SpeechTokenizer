# 三用户中心路由通信 Demo 使用说明

本目录是三用户中心路由实时语音通信第一版科研原型。

## 1. 文件说明

```text
router_server.py      中心路由，只转发密文 RVQ packet，不解密、不 decode、不混音
group_client.py       统一客户端，负责采集、encode、加密、接收、解密、decode、混音播放
common_protocol.py    TCP 消息封包工具，[4B header_len][header_json][body_bytes]
crypto_utils.py       XOR 演示性加解密工具
requirements.txt      最小依赖提示
```

默认加载 bundle 内自带的本地训练模型（即论文 SCIT-Speech-LCA v2，NAS 编码器 + LCA v2 权重，sha256=`af60223…`），key 复用两用户 demo 的密钥文件：

```text
../../../Log/spt_base/config.json
../../../Log/spt_base/SpeechTokenizer_best_dev.pt
../两用户通信demo/userA/46.txt
```

A/B/C 三端必须使用同一套模型、同一 `--rvq_layers`，否则接收端解码不匹配。

## 2. 单机自测

开 4 个 PowerShell 终端，全部 `cd` 进入本目录（`三用户中心路由通信demo/`），并先激活 conda 环境（见部署说明）。

终端 1 启动 Router：

```powershell
python router_server.py --listen_ip 127.0.0.1 --listen_port 12350 --room_id demo --max_queue 8
```

终端 2 启动 A，使用真实麦克风和耳机：

```powershell
python group_client.py --user_id A --room_id demo --router_ip 127.0.0.1 --router_port 12350 --mic_device 19 --spk_device 17 --device cpu --rvq_layers 3 --monitor
```

终端 3 启动 B，只接收和打印日志：

```powershell
python group_client.py --user_id B --room_id demo --router_ip 127.0.0.1 --router_port 12350 --no_mic --no_play --device cpu --rvq_layers 3 --monitor
```

终端 4 启动 C，只接收和打印日志：

```powershell
python group_client.py --user_id C --room_id demo --router_ip 127.0.0.1 --router_port 12350 --no_mic --no_play --device cpu --rvq_layers 3 --monitor
```

如果 A 说话时，Router 显示 forwarded，B/C 显示 `from_A` 的 seq、RMS、decode_ms，说明中心路由和多接收链路打通。

## 3. 单机混音测试

可以让 A 使用麦克风，B 使用 wav 文件模拟说话，C 负责播放混音。

示例：

```powershell
python group_client.py --user_id B --room_id demo --router_ip 127.0.0.1 --router_port 12350 --wav_input ..\..\..\example_input.wav --loop_wav --no_play --device cpu --rvq_layers 3 --monitor
```

C 去掉 `--no_play`：

```powershell
python group_client.py --user_id C --room_id demo --router_ip 127.0.0.1 --router_port 12350 --no_mic --spk_device 17 --device cpu --rvq_layers 3 --monitor
```

注意：`..\..\..\example_input.wav` 为 bundle 内自带测试音频；如用自己的 wav，替换为实际存在的路径即可。

## 4. 局域网三机测试

可以使用三台机器：

```text
机器 A: Router + Client A
机器 B: Client B
机器 C: Client C
```

机器 A 启动 Router：

```powershell
python router_server.py --listen_ip 0.0.0.0 --listen_port 12350 --room_id demo --max_queue 8
```

假设机器 A 局域网 IP 是 `192.168.31.39`，三台客户端都连接该 IP：

```powershell
python group_client.py --user_id A --room_id demo --router_ip 192.168.31.39 --router_port 12350 --mic_device <A麦克风ID> --spk_device <A耳机ID> --device cpu --rvq_layers 3 --monitor
python group_client.py --user_id B --room_id demo --router_ip 192.168.31.39 --router_port 12350 --mic_device <B麦克风ID> --spk_device <B耳机ID> --device cpu --rvq_layers 3 --monitor
python group_client.py --user_id C --room_id demo --router_ip 192.168.31.39 --router_port 12350 --mic_device <C麦克风ID> --spk_device <C耳机ID> --device cpu --rvq_layers 3 --monitor
```

机器 A 防火墙需要允许 TCP `12350` 端口。

## 5. 当前边界

第一版仍使用 TCP，适合局域网科研原型。弱网实时性后续建议升级 UDP/RTP 风格。

当前 XOR 是演示性密文转发，不等价于严格量子安全。若要声明量子安全，需要接入 QKD 密钥流、key offset 和消息认证。
