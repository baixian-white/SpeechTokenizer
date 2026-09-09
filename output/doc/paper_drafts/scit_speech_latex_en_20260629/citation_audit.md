# Citation Audit

Source draft: `output/doc/paper_drafts/scit_speech_method_cn_draft_20260622.md`

Status meanings:
- `verified`: title/authors/year and publication or official source were checked against a real source.
- `partially verified`: a real source was found, but one or more fields such as final venue, DOI, or complete author list should be manually checked before submission.
- `needs manual verification`: retained with minimal fields because I could not confirm the source sufficiently during this pass.

| key | paper/tool | status | verified source | notes |
|---|---|---|---|---|
| zeghidour2021soundstream | SoundStream: An End-to-End Neural Audio Codec | verified | arXiv + IEEE metadata | DOI included; check final formatting before submission. |
| defossez2022encodec | High Fidelity Neural Audio Compression | verified | arXiv | Kept as arXiv-safe metadata; no final venue asserted. |
| kumar2023dac | High-Fidelity Audio Compression with Improved RVQGAN | verified | Crossref + NeurIPS metadata | Formal NeurIPS 2023 metadata, pages, and DOI added. |
| zhang2023speechtokenizer | SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models | verified | arXiv | Title/authors/year checked from arXiv; no final venue asserted. |
| ji2024wavtokenizer | WavTokenizer: An Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling | verified | arXiv | Title/authors/year checked from arXiv; author-name typos corrected. |
| defossez2024moshi | Moshi: A Speech-Text Foundation Model for Real-Time Dialogue | partially verified | arXiv / Kyutai report page | Technical report entry, no DOI. |
| huang2023repcodec | RepCodec: A Speech Representation Codec for Speech Tokenization | verified | Crossref + ACL metadata | Formal ACL 2024 metadata, pages, and DOI added. |
| hsu2021hubert | HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units | verified | arXiv + IEEE metadata | DOI included. |
| borsos2022audiolm | AudioLM: A Language Modeling Approach to Audio Generation | verified | Crossref + IEEE metadata | Formal IEEE/ACM TASLP metadata and DOI added. |
| wang2023valle | Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers | verified | Crossref + IEEE metadata | Formal IEEE TASLP metadata and DOI added. |
| weng2021deepscs | Semantic Communication Systems for Speech Transmission | verified | Crossref + IEEE metadata | Formal IEEE JSAC metadata and DOI added. |
| qin2022semantic | Semantic Communications: Principles and Challenges | verified | arXiv | Added during latest source-draft resynchronization; no final venue asserted. |
| weng2022deepscst | Deep Learning Enabled Semantic Communications with Speech Recognition and Synthesis | verified | Crossref + IEEE metadata | Formal IEEE TWC metadata and DOI added. |
| tian2025largesc | Large Speech Model Enabled Semantic Communication | verified | arXiv | arXiv ID added and author names corrected. |
| han2025packetloss | Error-Resilient Semantic Communication for Speech Transmission over Packet-Loss Networks | verified | arXiv | Unsupported old prefix removed; arXiv ID added and author names corrected. |
| bourtsoulatze2018deepjscc | Deep Joint Source-Channel Coding for Wireless Image Transmission | verified | arXiv + IEEE metadata | DOI included. |
| valin2022plc | Real-Time Packet Loss Concealment with Mixed Generative and Predictive Model | verified | ISCA/Interspeech metadata | Pages and DOI checked. |
| westhausen2022tplcnet | tPLCnet: Real-time Deep Packet Loss Concealment in the Time Domain Using a Short Temporal Context | verified | ISCA/Interspeech metadata | Wrong pages corrected to 2903--2907; DOI added. |
| liu2018darts | DARTS: Differentiable Architecture Search | verified | arXiv + ICLR metadata | arXiv URL included. |
| cai2018proxylessnas | ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware | verified | arXiv + ICLR metadata | arXiv URL included. |
| luo2021lightspeech | LightSpeech: Lightweight and Fast Text to Speech with Neural Architecture Search | verified | IEEE/ICASSP metadata | DOI included. |
| radford2022whisper | Robust Speech Recognition via Large-Scale Weak Supervision | verified | arXiv + ICML/PMLR metadata | PMLR volume/pages added. |
| taal2011stoi | An Algorithm for Intelligibility Prediction of Time-Frequency Weighted Noisy Speech | verified | DOI / IEEE metadata | DOI included. |
| itu2001pesq | ITU-T P.862 PESQ recommendation | verified | ITU recommendation page | Recommendation entry, no DOI. |
| chinen2020visqol | ViSQOL v3 objective speech and audio metric | verified | Crossref + IEEE metadata | Formal QoMEX metadata and DOI added. |
| valin2012opus | Definition of the Opus Audio Codec | verified | RFC Editor | DOI and RFC URL included. |
| itu2002amrwb | ITU-T G.722.2 AMR-WB recommendation | verified | ITU recommendation page | Recommendation entry. |
| codec2repo | Codec 2 open source low bit rate speech codec | verified | official GitHub repository | Software citation only; replace with formal paper if required. |

All `\cite{}` keys used in `main.tex` are present in `references.bib` after the 2026-07-01 source-draft resynchronization.
