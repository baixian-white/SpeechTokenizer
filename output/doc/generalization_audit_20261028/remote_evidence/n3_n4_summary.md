# N3' and N4: matched CPU evaluation (L=3; 300 test-clean and 300 test-other utterances of exp12)

Means:

| split          | model    | condition     |   mel_l1 |   stoi |   pesq_wb |
|:---------------|:---------|:--------------|---------:|-------:|----------:|
| test-clean_300 | exp13    | clean         |   0.9721 | 0.8828 |    2.0071 |
| test-clean_300 | scit     | clean         |   0.8644 | 0.892  |    2.0976 |
| test-clean_300 | scit     | erasure10     |   0.9572 | 0.8526 |    1.6101 |
| test-clean_300 | scit     | substitution3 |   0.9328 | 0.8706 |    1.7961 |
| test-clean_300 | scit_lca | clean         |   0.8383 | 0.9    |    2.0972 |
| test-clean_300 | scit_lca | erasure10     |   0.9179 | 0.8664 |    1.6398 |
| test-clean_300 | scit_lca | substitution3 |   0.8983 | 0.8806 |    1.8292 |
| test-clean_300 | v0       | clean         |   0.7878 | 0.9076 |    2.2265 |
| test-clean_300 | v0       | erasure10     |   0.882  | 0.8695 |    1.6667 |
| test-clean_300 | v0       | substitution3 |   0.861  | 0.8863 |    1.8753 |
| test-other_300 | exp13    | clean         |   1.348  | 0.8424 |    1.6777 |
| test-other_300 | scit     | clean         |   1.1988 | 0.845  |    1.7558 |
| test-other_300 | scit     | erasure10     |   1.2712 | 0.8096 |    1.4525 |
| test-other_300 | scit     | substitution3 |   1.27   | 0.8251 |    1.5648 |
| test-other_300 | scit_lca | clean         |   1.1897 | 0.8573 |    1.7566 |
| test-other_300 | scit_lca | erasure10     |   1.2542 | 0.8273 |    1.4705 |
| test-other_300 | scit_lca | substitution3 |   1.2492 | 0.8391 |    1.593  |
| test-other_300 | v0       | clean         |   1.0895 | 0.8665 |    1.8754 |
| test-other_300 | v0       | erasure10     |   1.1644 | 0.8321 |    1.5163 |
| test-other_300 | v0       | substitution3 |   1.1674 | 0.8452 |    1.6451 |

Paired differences (first minus second; for mel-L1 negative favours the first). Intervals: 95%, utterance bootstrap and speaker-cluster bootstrap, 10,000 resamples, seed 42.

| split          | condition     | comparison                  | metric   |   n_utt |   n_spk |    mean |   utt_lo |   utt_hi |   spk_lo |   spk_hi |   speakers_favouring_first |
|:---------------|:--------------|:----------------------------|:---------|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------------------------:|
| test-clean_300 | clean         | scit_lca - scit             | mel_l1   |     300 |       6 | -0.026  |  -0.0343 |  -0.0172 |  -0.0596 |   0.0216 |                          5 |
| test-clean_300 | clean         | scit_lca - scit             | stoi     |     300 |       6 |  0.008  |   0.0073 |   0.0088 |   0.0063 |   0.0096 |                          6 |
| test-clean_300 | clean         | scit_lca - scit             | pesq_wb  |     300 |       6 | -0.0003 |  -0.016  |   0.0164 |  -0.0213 |   0.0205 |                          2 |
| test-clean_300 | clean         | v0 - scit                   | mel_l1   |     300 |       6 | -0.0766 |  -0.0797 |  -0.0736 |  -0.0856 |  -0.0669 |                          6 |
| test-clean_300 | clean         | v0 - scit                   | stoi     |     300 |       6 |  0.0156 |   0.0149 |   0.0164 |   0.0145 |   0.0169 |                          6 |
| test-clean_300 | clean         | v0 - scit                   | pesq_wb  |     300 |       6 |  0.1289 |   0.1142 |   0.1442 |   0.101  |   0.1527 |                          6 |
| test-clean_300 | clean         | scit_lca - v0               | mel_l1   |     300 |       6 |  0.0505 |   0.0439 |   0.058  |   0.0247 |   0.0883 |                          0 |
| test-clean_300 | clean         | scit_lca - v0               | stoi     |     300 |       6 | -0.0076 |  -0.0082 |  -0.0071 |  -0.0087 |  -0.0064 |                          0 |
| test-clean_300 | clean         | scit_lca - v0               | pesq_wb  |     300 |       6 | -0.1293 |  -0.1434 |  -0.1157 |  -0.1442 |  -0.1177 |                          0 |
| test-clean_300 | erasure10     | scit_lca - scit             | mel_l1   |     300 |       6 | -0.0393 |  -0.0475 |  -0.0303 |  -0.073  |   0.0074 |                          5 |
| test-clean_300 | erasure10     | scit_lca - scit             | stoi     |     300 |       6 |  0.0138 |   0.0128 |   0.0148 |   0.0119 |   0.0153 |                          6 |
| test-clean_300 | erasure10     | scit_lca - scit             | pesq_wb  |     300 |       6 |  0.0297 |   0.0188 |   0.0406 |   0.0188 |   0.0418 |                          6 |
| test-clean_300 | erasure10     | v0 - scit                   | mel_l1   |     300 |       6 | -0.0752 |  -0.0784 |  -0.0722 |  -0.0835 |  -0.0662 |                          6 |
| test-clean_300 | erasure10     | v0 - scit                   | stoi     |     300 |       6 |  0.0169 |   0.0159 |   0.0179 |   0.0153 |   0.0185 |                          6 |
| test-clean_300 | erasure10     | v0 - scit                   | pesq_wb  |     300 |       6 |  0.0566 |   0.0475 |   0.0657 |   0.049  |   0.067  |                          6 |
| test-clean_300 | erasure10     | scit_lca - v0               | mel_l1   |     300 |       6 |  0.0359 |   0.0288 |   0.0436 |   0.01   |   0.0734 |                          0 |
| test-clean_300 | erasure10     | scit_lca - v0               | stoi     |     300 |       6 | -0.0031 |  -0.0039 |  -0.0023 |  -0.0046 |  -0.0015 |                          0 |
| test-clean_300 | erasure10     | scit_lca - v0               | pesq_wb  |     300 |       6 | -0.0269 |  -0.0376 |  -0.0162 |  -0.0328 |  -0.0223 |                          0 |
| test-clean_300 | substitution3 | scit_lca - scit             | mel_l1   |     300 |       6 | -0.0345 |  -0.0424 |  -0.0259 |  -0.0671 |   0.0097 |                          5 |
| test-clean_300 | substitution3 | scit_lca - scit             | stoi     |     300 |       6 |  0.01   |   0.009  |   0.0111 |   0.0085 |   0.0114 |                          6 |
| test-clean_300 | substitution3 | scit_lca - scit             | pesq_wb  |     300 |       6 |  0.033  |   0.0163 |   0.0507 |   0.0117 |   0.0592 |                          5 |
| test-clean_300 | substitution3 | v0 - scit                   | mel_l1   |     300 |       6 | -0.0718 |  -0.0752 |  -0.0684 |  -0.0828 |  -0.0588 |                          6 |
| test-clean_300 | substitution3 | v0 - scit                   | stoi     |     300 |       6 |  0.0157 |   0.0148 |   0.0166 |   0.0148 |   0.0166 |                          6 |
| test-clean_300 | substitution3 | v0 - scit                   | pesq_wb  |     300 |       6 |  0.0792 |   0.0654 |   0.0931 |   0.0607 |   0.0957 |                          6 |
| test-clean_300 | substitution3 | scit_lca - v0               | mel_l1   |     300 |       6 |  0.0373 |   0.0308 |   0.0441 |   0.0149 |   0.069  |                          0 |
| test-clean_300 | substitution3 | scit_lca - v0               | stoi     |     300 |       6 | -0.0057 |  -0.0066 |  -0.0048 |  -0.0071 |  -0.0042 |                          0 |
| test-clean_300 | substitution3 | scit_lca - v0               | pesq_wb  |     300 |       6 | -0.0461 |  -0.0629 |  -0.0282 |  -0.0651 |  -0.0301 |                          0 |
| test-clean_300 | clean         | exp13 - scit                | mel_l1   |     300 |       6 |  0.1078 |   0.0973 |   0.1184 |   0.0581 |   0.1685 |                          0 |
| test-clean_300 | clean         | exp13 - scit                | stoi     |     300 |       6 | -0.0092 |  -0.01   |  -0.0084 |  -0.0107 |  -0.0078 |                          0 |
| test-clean_300 | clean         | exp13 - scit                | pesq_wb  |     300 |       6 | -0.0905 |  -0.1061 |  -0.0744 |  -0.1256 |  -0.0589 |                          0 |
| test-clean_300 | erasure10     | degradation scit_lca - v0   | mel_l1   |     300 |       6 | -0.0147 |  -0.0162 |  -0.0132 |  -0.0159 |  -0.0138 |                          6 |
| test-clean_300 | erasure10     | degradation scit_lca - scit | mel_l1   |     300 |       6 | -0.0133 |  -0.0149 |  -0.0117 |  -0.014  |  -0.0123 |                          6 |
| test-clean_300 | erasure10     | degradation v0 - scit       | mel_l1   |     300 |       6 |  0.0014 |   0      |   0.0028 |   0.0004 |   0.003  |                          1 |
| test-clean_300 | substitution3 | degradation scit_lca - v0   | mel_l1   |     300 |       6 | -0.0133 |  -0.0169 |  -0.0098 |  -0.0206 |  -0.0074 |                          6 |
| test-clean_300 | substitution3 | degradation scit_lca - scit | mel_l1   |     300 |       6 | -0.0085 |  -0.0121 |  -0.0047 |  -0.0128 |  -0.0049 |                          6 |
| test-clean_300 | substitution3 | degradation v0 - scit       | mel_l1   |     300 |       6 |  0.0048 |   0.0034 |   0.0064 |   0.0021 |   0.0085 |                          0 |
| test-other_300 | clean         | scit_lca - scit             | mel_l1   |     300 |       4 | -0.0091 |  -0.0145 |  -0.0035 |  -0.0276 |   0.006  |                          1 |
| test-other_300 | clean         | scit_lca - scit             | stoi     |     300 |       4 |  0.0123 |   0.0109 |   0.0137 |   0.0027 |   0.0177 |                          4 |
| test-other_300 | clean         | scit_lca - scit             | pesq_wb  |     300 |       4 |  0.0007 |  -0.0111 |   0.0131 |  -0.0744 |   0.0468 |                          1 |
| test-other_300 | clean         | v0 - scit                   | mel_l1   |     300 |       4 | -0.1093 |  -0.1164 |  -0.1027 |  -0.159  |  -0.0732 |                          4 |
| test-other_300 | clean         | v0 - scit                   | stoi     |     300 |       4 |  0.0215 |   0.0202 |   0.0228 |   0.0141 |   0.0263 |                          4 |
| test-other_300 | clean         | v0 - scit                   | pesq_wb  |     300 |       4 |  0.1196 |   0.109  |   0.1302 |   0.079  |   0.1525 |                          4 |
| test-other_300 | clean         | scit_lca - v0               | mel_l1   |     300 |       4 |  0.1002 |   0.0909 |   0.1099 |   0.0531 |   0.165  |                          0 |
| test-other_300 | clean         | scit_lca - v0               | stoi     |     300 |       4 | -0.0092 |  -0.0102 |  -0.0081 |  -0.0114 |  -0.0082 |                          0 |
| test-other_300 | clean         | scit_lca - v0               | pesq_wb  |     300 |       4 | -0.1188 |  -0.1299 |  -0.108  |  -0.1713 |  -0.0933 |                          0 |
| test-other_300 | erasure10     | scit_lca - scit             | mel_l1   |     300 |       4 | -0.017  |  -0.0228 |  -0.0112 |  -0.037  |  -0.0001 |                          3 |
| test-other_300 | erasure10     | scit_lca - scit             | stoi     |     300 |       4 |  0.0177 |   0.0162 |   0.0192 |   0.0097 |   0.0211 |                          4 |
| test-other_300 | erasure10     | scit_lca - scit             | pesq_wb  |     300 |       4 |  0.018  |   0.0079 |   0.0281 |  -0.0208 |   0.0541 |                          1 |
| test-other_300 | erasure10     | v0 - scit                   | mel_l1   |     300 |       4 | -0.1068 |  -0.1141 |  -0.1    |  -0.155  |  -0.0704 |                          4 |
| test-other_300 | erasure10     | v0 - scit                   | stoi     |     300 |       4 |  0.0225 |   0.0211 |   0.0239 |   0.0161 |   0.0268 |                          4 |
| test-other_300 | erasure10     | v0 - scit                   | pesq_wb  |     300 |       4 |  0.0639 |   0.0553 |   0.0728 |   0.0473 |   0.0815 |                          4 |
| test-other_300 | erasure10     | scit_lca - v0               | mel_l1   |     300 |       4 |  0.0898 |   0.0803 |   0.0999 |   0.0427 |   0.1545 |                          0 |
| test-other_300 | erasure10     | scit_lca - v0               | stoi     |     300 |       4 | -0.0048 |  -0.0061 |  -0.0035 |  -0.0074 |  -0.0031 |                          0 |
| test-other_300 | erasure10     | scit_lca - v0               | pesq_wb  |     300 |       4 | -0.0459 |  -0.0558 |  -0.0363 |  -0.0708 |  -0.0274 |                          0 |
| test-other_300 | substitution3 | scit_lca - scit             | mel_l1   |     300 |       4 | -0.0208 |  -0.0265 |  -0.0148 |  -0.0402 |   0.001  |                          3 |
| test-other_300 | substitution3 | scit_lca - scit             | stoi     |     300 |       4 |  0.0139 |   0.0125 |   0.0154 |   0.0043 |   0.0181 |                          4 |
| test-other_300 | substitution3 | scit_lca - scit             | pesq_wb  |     300 |       4 |  0.0282 |   0.0164 |   0.0399 |  -0.0207 |   0.0713 |                          2 |
| test-other_300 | substitution3 | v0 - scit                   | mel_l1   |     300 |       4 | -0.1026 |  -0.1097 |  -0.0959 |  -0.1496 |  -0.0616 |                          4 |
| test-other_300 | substitution3 | v0 - scit                   | stoi     |     300 |       4 |  0.0201 |   0.0188 |   0.0213 |   0.0143 |   0.0243 |                          4 |
| test-other_300 | substitution3 | v0 - scit                   | pesq_wb  |     300 |       4 |  0.0803 |   0.0697 |   0.0914 |   0.0484 |   0.1063 |                          4 |
| test-other_300 | substitution3 | scit_lca - v0               | mel_l1   |     300 |       4 |  0.0818 |   0.0722 |   0.0921 |   0.0349 |   0.1506 |                          0 |
| test-other_300 | substitution3 | scit_lca - v0               | stoi     |     300 |       4 | -0.0061 |  -0.0073 |  -0.005  |  -0.0099 |  -0.0037 |                          0 |
| test-other_300 | substitution3 | scit_lca - v0               | pesq_wb  |     300 |       4 | -0.0521 |  -0.0636 |  -0.0405 |  -0.0826 |  -0.0332 |                          0 |
| test-other_300 | clean         | exp13 - scit                | mel_l1   |     300 |       4 |  0.1492 |   0.1317 |   0.1672 |   0.0243 |   0.2809 |                          1 |
| test-other_300 | clean         | exp13 - scit                | stoi     |     300 |       4 | -0.0027 |  -0.004  |  -0.0013 |  -0.0095 |   0.0002 |                          1 |
| test-other_300 | clean         | exp13 - scit                | pesq_wb  |     300 |       4 | -0.0781 |  -0.0912 |  -0.0654 |  -0.1373 |  -0.0338 |                          1 |
| test-other_300 | erasure10     | degradation scit_lca - v0   | mel_l1   |     300 |       4 | -0.0104 |  -0.0124 |  -0.0085 |  -0.0125 |  -0.0077 |                          4 |
| test-other_300 | erasure10     | degradation scit_lca - scit | mel_l1   |     300 |       4 | -0.0079 |  -0.0099 |  -0.0059 |  -0.0093 |  -0.0061 |                          4 |
| test-other_300 | erasure10     | degradation v0 - scit       | mel_l1   |     300 |       4 |  0.0025 |   0.0007 |   0.0043 |   0.0005 |   0.0042 |                          1 |
| test-other_300 | substitution3 | degradation scit_lca - v0   | mel_l1   |     300 |       4 | -0.0185 |  -0.0217 |  -0.0154 |  -0.0341 |  -0.0117 |                          4 |
| test-other_300 | substitution3 | degradation scit_lca - scit | mel_l1   |     300 |       4 | -0.0117 |  -0.0147 |  -0.0088 |  -0.0225 |  -0.0048 |                          4 |
| test-other_300 | substitution3 | degradation v0 - scit       | mel_l1   |     300 |       4 |  0.0068 |   0.0049 |   0.0087 |   0.0019 |   0.0148 |                          0 |
