# Decoder Condition

- Main NAS search target: transmitter-side encoder before latent Z.
- Teacher-guided proxy condition: frozen pretrained SpeechTokenizer transform/RVQ/decoder.
- Candidate decoder ops, width, depth, activation, and LSTM settings are not searched.
- When teacher guidance is enabled, candidate encoders are evaluated by plugging into the frozen pretrained downstream components.
