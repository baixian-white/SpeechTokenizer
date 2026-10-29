# Component attribution of channel-aware fine-tuning (CPU; L=3; 300 test-clean + 300 test-other utterances)

Means (columns: models):

|                                     |   ('mel_l1', 'scit') |   ('mel_l1', 'v0') |   ('mel_l1', 'v1') |   ('mel_l1', 'v2') |   ('mel_l1', 'v3') |   ('mel_l1', 'v4') |   ('stoi', 'scit') |   ('stoi', 'v0') |   ('stoi', 'v1') |   ('stoi', 'v2') |   ('stoi', 'v3') |   ('stoi', 'v4') |   ('pesq_wb', 'scit') |   ('pesq_wb', 'v0') |   ('pesq_wb', 'v1') |   ('pesq_wb', 'v2') |   ('pesq_wb', 'v3') |   ('pesq_wb', 'v4') |
|:------------------------------------|---------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-----------------:|-----------------:|-----------------:|-----------------:|-----------------:|----------------------:|--------------------:|--------------------:|--------------------:|--------------------:|--------------------:|
| ('test-clean_300', 'clean')         |               0.8644 |             0.7878 |             0.8356 |             0.8525 |             0.8327 |             0.8383 |             0.892  |           0.9076 |           0.9012 |           0.8956 |           0.9011 |           0.9    |                2.0976 |              2.2265 |              2.134  |              2.0841 |              2.119  |              2.0972 |
| ('test-clean_300', 'erasure10')     |               0.9572 |             0.882  |             0.9252 |             0.9344 |             0.9155 |             0.9179 |             0.8526 |           0.8695 |           0.8649 |           0.8613 |           0.8663 |           0.8664 |                1.6101 |              1.6667 |              1.6316 |              1.6362 |              1.647  |              1.6398 |
| ('test-clean_300', 'substitution3') |               0.9328 |             0.861  |             0.9239 |             0.9251 |             0.8953 |             0.8983 |             0.8706 |           0.8863 |           0.8759 |           0.8724 |           0.8793 |           0.8806 |                1.7961 |              1.8753 |              1.7739 |              1.7944 |              1.8297 |              1.8292 |
| ('test-other_300', 'clean')         |               1.1988 |             1.0895 |             1.1832 |             1.2199 |             1.1692 |             1.1897 |             0.845  |           0.8665 |           0.858  |           0.8507 |           0.8592 |           0.8573 |                1.7558 |              1.8754 |              1.7979 |              1.7395 |              1.7756 |              1.7566 |
| ('test-other_300', 'erasure10')     |               1.2712 |             1.1644 |             1.2549 |             1.2866 |             1.2363 |             1.2542 |             0.8096 |           0.8321 |           0.8259 |           0.8195 |           0.828  |           0.8273 |                1.4525 |              1.5163 |              1.4855 |              1.4646 |              1.4856 |              1.4705 |
| ('test-other_300', 'substitution3') |               1.27   |             1.1674 |             1.2709 |             1.2957 |             1.2303 |             1.2492 |             0.8251 |           0.8452 |           0.8343 |           0.828  |           0.8394 |           0.8391 |                1.5648 |              1.6451 |              1.5767 |              1.5606 |              1.6043 |              1.593  |

Mel-L1 degradation from each model's own clean level:

|                                     |   scit |     v0 |     v1 |     v2 |     v3 |     v4 |
|:------------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|
| ('test-clean_300', 'erasure10')     | 0.0929 | 0.0943 | 0.0897 | 0.0819 | 0.0828 | 0.0796 |
| ('test-clean_300', 'substitution3') | 0.0685 | 0.0733 | 0.0883 | 0.0727 | 0.0626 | 0.06   |
| ('test-other_300', 'erasure10')     | 0.0723 | 0.0749 | 0.0717 | 0.0667 | 0.0671 | 0.0644 |
| ('test-other_300', 'substitution3') | 0.0711 | 0.0779 | 0.0877 | 0.0758 | 0.0611 | 0.0594 |

Contrasts (first minus second; speaker-cluster 95% intervals; for mel-L1 and degradation negative favours the first):

| split          | condition     | contrast   | component                        | metric          |    mean |   spk_lo |   spk_hi |   n_spk |   speakers_favouring_first |
|:---------------|:--------------|:-----------|:---------------------------------|:----------------|--------:|---------:|---------:|--------:|---------------------------:|
| test-clean_300 | clean         | v2 - v0    | channel simulation, fixed depth  | mel_l1          |  0.0647 |   0.0394 |   0.0999 |       6 |                          0 |
| test-clean_300 | clean         | v2 - v0    | channel simulation, fixed depth  | stoi            | -0.012  |  -0.0138 |  -0.0104 |       6 |                          0 |
| test-clean_300 | clean         | v2 - v0    | channel simulation, fixed depth  | pesq_wb         | -0.1424 |  -0.1594 |  -0.1267 |       6 |                          0 |
| test-clean_300 | clean         | v3 - v1    | channel simulation, random depth | mel_l1          | -0.0028 |  -0.0103 |   0.0041 |       6 |                          5 |
| test-clean_300 | clean         | v3 - v1    | channel simulation, random depth | stoi            | -0.0001 |  -0.0007 |   0.0005 |       6 |                          3 |
| test-clean_300 | clean         | v3 - v1    | channel simulation, random depth | pesq_wb         | -0.015  |  -0.0261 |  -0.0022 |       6 |                          1 |
| test-clean_300 | clean         | v4 - v3    | consistency loss                 | mel_l1          |  0.0056 |   0.0019 |   0.0093 |       6 |                          1 |
| test-clean_300 | clean         | v4 - v3    | consistency loss                 | stoi            | -0.0011 |  -0.0015 |  -0.0005 |       6 |                          1 |
| test-clean_300 | clean         | v4 - v3    | consistency loss                 | pesq_wb         | -0.0217 |  -0.0341 |  -0.0096 |       6 |                          0 |
| test-clean_300 | clean         | v1 - v0    | random depth                     | mel_l1          |  0.0478 |   0.0301 |   0.0774 |       6 |                          0 |
| test-clean_300 | clean         | v1 - v0    | random depth                     | stoi            | -0.0065 |  -0.0079 |  -0.0049 |       6 |                          0 |
| test-clean_300 | clean         | v1 - v0    | random depth                     | pesq_wb         | -0.0925 |  -0.1037 |  -0.0822 |       6 |                          0 |
| test-clean_300 | clean         | v4 - v0    | all three (SCIT-LCA - V0)        | mel_l1          |  0.0505 |   0.0247 |   0.0883 |       6 |                          0 |
| test-clean_300 | clean         | v4 - v0    | all three (SCIT-LCA - V0)        | stoi            | -0.0076 |  -0.0088 |  -0.0064 |       6 |                          0 |
| test-clean_300 | clean         | v4 - v0    | all three (SCIT-LCA - V0)        | pesq_wb         | -0.1293 |  -0.1434 |  -0.1178 |       6 |                          0 |
| test-clean_300 | erasure10     | v2 - v0    | channel simulation, fixed depth  | mel_l1          |  0.0523 |   0.0286 |   0.0869 |       6 |                          0 |
| test-clean_300 | erasure10     | v2 - v0    | channel simulation, fixed depth  | stoi            | -0.0082 |  -0.0105 |  -0.006  |       6 |                          0 |
| test-clean_300 | erasure10     | v2 - v0    | channel simulation, fixed depth  | pesq_wb         | -0.0304 |  -0.0432 |  -0.0142 |       6 |                          1 |
| test-clean_300 | erasure10     | v2 - v0    | channel simulation, fixed depth  | mel_degradation | -0.0123 |  -0.0145 |  -0.0102 |       6 |                          6 |
| test-clean_300 | erasure10     | v3 - v1    | channel simulation, random depth | mel_l1          | -0.0097 |  -0.0175 |  -0.0024 |       6 |                          5 |
| test-clean_300 | erasure10     | v3 - v1    | channel simulation, random depth | stoi            |  0.0014 |   0.0004 |   0.0023 |       6 |                          5 |
| test-clean_300 | erasure10     | v3 - v1    | channel simulation, random depth | pesq_wb         |  0.0153 |   0.0088 |   0.022  |       6 |                          5 |
| test-clean_300 | erasure10     | v3 - v1    | channel simulation, random depth | mel_degradation | -0.0069 |  -0.0083 |  -0.0049 |       6 |                          6 |
| test-clean_300 | erasure10     | v4 - v3    | consistency loss                 | mel_l1          |  0.0024 |  -0.002  |   0.0062 |       6 |                          2 |
| test-clean_300 | erasure10     | v4 - v3    | consistency loss                 | stoi            |  0.0001 |  -0.0008 |   0.001  |       6 |                          4 |
| test-clean_300 | erasure10     | v4 - v3    | consistency loss                 | pesq_wb         | -0.0072 |  -0.0188 |   0.0022 |       6 |                          3 |
| test-clean_300 | erasure10     | v4 - v3    | consistency loss                 | mel_degradation | -0.0032 |  -0.0042 |  -0.0022 |       6 |                          6 |
| test-clean_300 | erasure10     | v1 - v0    | random depth                     | mel_l1          |  0.0432 |   0.0246 |   0.0715 |       6 |                          0 |
| test-clean_300 | erasure10     | v1 - v0    | random depth                     | stoi            | -0.0046 |  -0.007  |  -0.002  |       6 |                          0 |
| test-clean_300 | erasure10     | v1 - v0    | random depth                     | pesq_wb         | -0.035  |  -0.0491 |  -0.0187 |       6 |                          0 |
| test-clean_300 | erasure10     | v1 - v0    | random depth                     | mel_degradation | -0.0046 |  -0.0073 |  -0.0021 |       6 |                          5 |
| test-clean_300 | erasure10     | v4 - v0    | all three (SCIT-LCA - V0)        | mel_l1          |  0.0359 |   0.01   |   0.0734 |       6 |                          0 |
| test-clean_300 | erasure10     | v4 - v0    | all three (SCIT-LCA - V0)        | stoi            | -0.0031 |  -0.0046 |  -0.0015 |       6 |                          0 |
| test-clean_300 | erasure10     | v4 - v0    | all three (SCIT-LCA - V0)        | pesq_wb         | -0.0269 |  -0.0328 |  -0.0222 |       6 |                          0 |
| test-clean_300 | erasure10     | v4 - v0    | all three (SCIT-LCA - V0)        | mel_degradation | -0.0147 |  -0.0159 |  -0.0137 |       6 |                          6 |
| test-clean_300 | substitution3 | v2 - v0    | channel simulation, fixed depth  | mel_l1          |  0.0641 |   0.0442 |   0.0942 |       6 |                          0 |
| test-clean_300 | substitution3 | v2 - v0    | channel simulation, fixed depth  | stoi            | -0.0139 |  -0.0156 |  -0.0124 |       6 |                          0 |
| test-clean_300 | substitution3 | v2 - v0    | channel simulation, fixed depth  | pesq_wb         | -0.0809 |  -0.0968 |  -0.0613 |       6 |                          0 |
| test-clean_300 | substitution3 | v2 - v0    | channel simulation, fixed depth  | mel_degradation | -0.0006 |  -0.0071 |   0.0056 |       6 |                          3 |
| test-clean_300 | substitution3 | v3 - v1    | channel simulation, random depth | mel_l1          | -0.0285 |  -0.0404 |  -0.0183 |       6 |                          6 |
| test-clean_300 | substitution3 | v3 - v1    | channel simulation, random depth | stoi            |  0.0033 |   0.0025 |   0.0042 |       6 |                          6 |
| test-clean_300 | substitution3 | v3 - v1    | channel simulation, random depth | pesq_wb         |  0.0558 |   0.0406 |   0.0689 |       6 |                          6 |
| test-clean_300 | substitution3 | v3 - v1    | channel simulation, random depth | mel_degradation | -0.0257 |  -0.0433 |  -0.0142 |       6 |                          6 |
| test-clean_300 | substitution3 | v4 - v3    | consistency loss                 | mel_l1          |  0.003  |  -0.0016 |   0.0071 |       6 |                          2 |
| test-clean_300 | substitution3 | v4 - v3    | consistency loss                 | stoi            |  0.0013 |   0.0005 |   0.0021 |       6 |                          5 |
| test-clean_300 | substitution3 | v4 - v3    | consistency loss                 | pesq_wb         | -0.0006 |  -0.0138 |   0.0147 |       6 |                          3 |
| test-clean_300 | substitution3 | v4 - v3    | consistency loss                 | mel_degradation | -0.0026 |  -0.0042 |  -0.0012 |       6 |                          6 |
| test-clean_300 | substitution3 | v1 - v0    | random depth                     | mel_l1          |  0.0628 |   0.0394 |   0.1009 |       6 |                          0 |
| test-clean_300 | substitution3 | v1 - v0    | random depth                     | stoi            | -0.0104 |  -0.012  |  -0.0087 |       6 |                          0 |
| test-clean_300 | substitution3 | v1 - v0    | random depth                     | pesq_wb         | -0.1014 |  -0.1203 |  -0.0855 |       6 |                          0 |
| test-clean_300 | substitution3 | v1 - v0    | random depth                     | mel_degradation |  0.015  |   0.0075 |   0.0251 |       6 |                          0 |
| test-clean_300 | substitution3 | v4 - v0    | all three (SCIT-LCA - V0)        | mel_l1          |  0.0373 |   0.0145 |   0.069  |       6 |                          0 |
| test-clean_300 | substitution3 | v4 - v0    | all three (SCIT-LCA - V0)        | stoi            | -0.0057 |  -0.0071 |  -0.0042 |       6 |                          0 |
| test-clean_300 | substitution3 | v4 - v0    | all three (SCIT-LCA - V0)        | pesq_wb         | -0.0461 |  -0.0652 |  -0.0304 |       6 |                          0 |
| test-clean_300 | substitution3 | v4 - v0    | all three (SCIT-LCA - V0)        | mel_degradation | -0.0133 |  -0.0207 |  -0.0074 |       6 |                          6 |
| test-other_300 | clean         | v2 - v0    | channel simulation, fixed depth  | mel_l1          |  0.1304 |   0.0835 |   0.1889 |       4 |                          0 |
| test-other_300 | clean         | v2 - v0    | channel simulation, fixed depth  | stoi            | -0.0158 |  -0.0213 |  -0.0124 |       4 |                          0 |
| test-other_300 | clean         | v2 - v0    | channel simulation, fixed depth  | pesq_wb         | -0.1359 |  -0.2007 |  -0.0846 |       4 |                          0 |
| test-other_300 | clean         | v3 - v1    | channel simulation, random depth | mel_l1          | -0.014  |  -0.0269 |  -0.001  |       4 |                          3 |
| test-other_300 | clean         | v3 - v1    | channel simulation, random depth | stoi            |  0.0012 |   0.0008 |   0.0016 |       4 |                          4 |
| test-other_300 | clean         | v3 - v1    | channel simulation, random depth | pesq_wb         | -0.0223 |  -0.0347 |  -0.0043 |       4 |                          1 |
| test-other_300 | clean         | v4 - v3    | consistency loss                 | mel_l1          |  0.0205 |   0.0068 |   0.0377 |       4 |                          0 |
| test-other_300 | clean         | v4 - v3    | consistency loss                 | stoi            | -0.0019 |  -0.0024 |  -0.0017 |       4 |                          0 |
| test-other_300 | clean         | v4 - v3    | consistency loss                 | pesq_wb         | -0.019  |  -0.0334 |  -0.0108 |       4 |                          0 |
| test-other_300 | clean         | v1 - v0    | random depth                     | mel_l1          |  0.0937 |   0.0674 |   0.1283 |       4 |                          0 |
| test-other_300 | clean         | v1 - v0    | random depth                     | stoi            | -0.0085 |  -0.0107 |  -0.0074 |       4 |                          0 |
| test-other_300 | clean         | v1 - v0    | random depth                     | pesq_wb         | -0.0775 |  -0.1348 |  -0.0533 |       4 |                          0 |
| test-other_300 | clean         | v4 - v0    | all three (SCIT-LCA - V0)        | mel_l1          |  0.1002 |   0.0531 |   0.165  |       4 |                          0 |
| test-other_300 | clean         | v4 - v0    | all three (SCIT-LCA - V0)        | stoi            | -0.0092 |  -0.0114 |  -0.0082 |       4 |                          0 |
| test-other_300 | clean         | v4 - v0    | all three (SCIT-LCA - V0)        | pesq_wb         | -0.1188 |  -0.1713 |  -0.0933 |       4 |                          0 |
| test-other_300 | erasure10     | v2 - v0    | channel simulation, fixed depth  | mel_l1          |  0.1222 |   0.072  |   0.1825 |       4 |                          0 |
| test-other_300 | erasure10     | v2 - v0    | channel simulation, fixed depth  | stoi            | -0.0126 |  -0.0173 |  -0.0088 |       4 |                          0 |
| test-other_300 | erasure10     | v2 - v0    | channel simulation, fixed depth  | pesq_wb         | -0.0518 |  -0.0673 |  -0.0428 |       4 |                          0 |
| test-other_300 | erasure10     | v2 - v0    | channel simulation, fixed depth  | mel_degradation | -0.0082 |  -0.0115 |  -0.0056 |       4 |                          4 |
| test-other_300 | erasure10     | v3 - v1    | channel simulation, random depth | mel_l1          | -0.0185 |  -0.0329 |  -0.0049 |       4 |                          4 |
| test-other_300 | erasure10     | v3 - v1    | channel simulation, random depth | stoi            |  0.0022 |   0.0012 |   0.0034 |       4 |                          4 |
| test-other_300 | erasure10     | v3 - v1    | channel simulation, random depth | pesq_wb         |  0.0001 |  -0.0084 |   0.0078 |       4 |                          2 |
| test-other_300 | erasure10     | v3 - v1    | channel simulation, random depth | mel_degradation | -0.0045 |  -0.0057 |  -0.0024 |       4 |                          4 |
| test-other_300 | erasure10     | v4 - v3    | consistency loss                 | mel_l1          |  0.0178 |   0.0051 |   0.034  |       4 |                          0 |
| test-other_300 | erasure10     | v4 - v3    | consistency loss                 | stoi            | -0.0007 |  -0.0009 |  -0.0002 |       4 |                          0 |
| test-other_300 | erasure10     | v4 - v3    | consistency loss                 | pesq_wb         | -0.0151 |  -0.0186 |  -0.0101 |       4 |                          0 |
| test-other_300 | erasure10     | v4 - v3    | consistency loss                 | mel_degradation | -0.0027 |  -0.0054 |  -0.001  |       4 |                          4 |
| test-other_300 | erasure10     | v1 - v0    | random depth                     | mel_l1          |  0.0905 |   0.0643 |   0.1254 |       4 |                          0 |
| test-other_300 | erasure10     | v1 - v0    | random depth                     | stoi            | -0.0062 |  -0.009  |  -0.0045 |       4 |                          0 |
| test-other_300 | erasure10     | v1 - v0    | random depth                     | pesq_wb         | -0.0309 |  -0.0624 |  -0.0175 |       4 |                          0 |
| test-other_300 | erasure10     | v1 - v0    | random depth                     | mel_degradation | -0.0032 |  -0.0037 |  -0.0026 |       4 |                          4 |
| test-other_300 | erasure10     | v4 - v0    | all three (SCIT-LCA - V0)        | mel_l1          |  0.0898 |   0.0427 |   0.1545 |       4 |                          0 |
| test-other_300 | erasure10     | v4 - v0    | all three (SCIT-LCA - V0)        | stoi            | -0.0048 |  -0.0074 |  -0.0031 |       4 |                          0 |
| test-other_300 | erasure10     | v4 - v0    | all three (SCIT-LCA - V0)        | pesq_wb         | -0.0459 |  -0.0708 |  -0.0274 |       4 |                          0 |
| test-other_300 | erasure10     | v4 - v0    | all three (SCIT-LCA - V0)        | mel_degradation | -0.0104 |  -0.0125 |  -0.0077 |       4 |                          4 |
| test-other_300 | substitution3 | v2 - v0    | channel simulation, fixed depth  | mel_l1          |  0.1283 |   0.0821 |   0.1895 |       4 |                          0 |
| test-other_300 | substitution3 | v2 - v0    | channel simulation, fixed depth  | stoi            | -0.0172 |  -0.0246 |  -0.013  |       4 |                          0 |
| test-other_300 | substitution3 | v2 - v0    | channel simulation, fixed depth  | pesq_wb         | -0.0845 |  -0.124  |  -0.0565 |       4 |                          0 |
| test-other_300 | substitution3 | v2 - v0    | channel simulation, fixed depth  | mel_degradation | -0.0021 |  -0.0058 |   0.0014 |       4 |                          2 |
| test-other_300 | substitution3 | v3 - v1    | channel simulation, random depth | mel_l1          | -0.0407 |  -0.0708 |  -0.0223 |       4 |                          4 |
| test-other_300 | substitution3 | v3 - v1    | channel simulation, random depth | stoi            |  0.0051 |   0.0029 |   0.0066 |       4 |                          4 |
| test-other_300 | substitution3 | v3 - v1    | channel simulation, random depth | pesq_wb         |  0.0276 |   0.0085 |   0.043  |       4 |                          4 |
| test-other_300 | substitution3 | v3 - v1    | channel simulation, random depth | mel_degradation | -0.0267 |  -0.0612 |  -0.0121 |       4 |                          4 |
| test-other_300 | substitution3 | v4 - v3    | consistency loss                 | mel_l1          |  0.0189 |   0.0055 |   0.0369 |       4 |                          0 |
| test-other_300 | substitution3 | v4 - v3    | consistency loss                 | stoi            | -0.0003 |  -0.001  |   0      |       4 |                          1 |
| test-other_300 | substitution3 | v4 - v3    | consistency loss                 | pesq_wb         | -0.0113 |  -0.0193 |  -0.0059 |       4 |                          0 |
| test-other_300 | substitution3 | v4 - v3    | consistency loss                 | mel_degradation | -0.0016 |  -0.0052 |   0.0001 |       4 |                          3 |
| test-other_300 | substitution3 | v1 - v0    | random depth                     | mel_l1          |  0.1035 |   0.0744 |   0.1382 |       4 |                          0 |
| test-other_300 | substitution3 | v1 - v0    | random depth                     | stoi            | -0.0109 |  -0.012  |  -0.0102 |       4 |                          0 |
| test-other_300 | substitution3 | v1 - v0    | random depth                     | pesq_wb         | -0.0683 |  -0.0939 |  -0.051  |       4 |                          0 |
| test-other_300 | substitution3 | v1 - v0    | random depth                     | mel_degradation |  0.0098 |  -0.0003 |   0.0323 |       4 |                          1 |
| test-other_300 | substitution3 | v4 - v0    | all three (SCIT-LCA - V0)        | mel_l1          |  0.0818 |   0.0349 |   0.1506 |       4 |                          0 |
| test-other_300 | substitution3 | v4 - v0    | all three (SCIT-LCA - V0)        | stoi            | -0.0061 |  -0.0099 |  -0.0037 |       4 |                          0 |
| test-other_300 | substitution3 | v4 - v0    | all three (SCIT-LCA - V0)        | pesq_wb         | -0.0521 |  -0.0826 |  -0.0332 |       4 |                          0 |
| test-other_300 | substitution3 | v4 - v0    | all three (SCIT-LCA - V0)        | mel_degradation | -0.0185 |  -0.0341 |  -0.0117 |       4 |                          4 |
