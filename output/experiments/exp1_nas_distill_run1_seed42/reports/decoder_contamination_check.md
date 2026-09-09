# Decoder Contamination Check

- The staged search space contains encoder macro/block choices only.
- RVQ codebooks, M, K, L, payload accounting, and ChannelSim are not searched in this script.
- Short distillation updates only the NAS candidate encoder.
- Proxy reconstruction uses the frozen pretrained SpeechTokenizer decoder when teacher guidance is enabled.
- Encoder complexity metrics report encoder-only params, MACs, and RTF.
