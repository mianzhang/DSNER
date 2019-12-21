# Performance Comparision

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H|EC-Dev|63.78|61.26|62.49
This Implementation(LSTM-CRF)|H|EC-Dev|**65.14**|59.79|62.35

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H|EC-Test|59.93|58.46|59.19
This Implementation(LSTM-CRF)|H|EC-Test|**62.81**|57.41|**59.99**


Model|Training Data|Dev|Test
-|:-:|:-:|:-:
LSTM-CRF|H|63.78(P) 61.26(R) 62.49(F1)|59.93(P) 58.46(R) 59.19(F1)
This Implementation(LSTM-CRF)|H|65.14(P) 59.79(R) 62.35(F1)|62.81(P) 57.41(R) 59.99(F1)
LSTM-CRF|H + A|67.75(P) 52.91(R) 59.42(F1)|62.36(P) 48.54(R) 54.59(F1)
This Implementation(LSTM-CRF)|H + A|69.27(P) 54.11(R) 60.76(F1)|65.77(P) 50.44(R) 57.09(F1)
LSTM-CRF-PA|H + A|60.34(P) 64.49(R) 62.35(F1)|59.36(P) 60.82(R) 60.08(F1)
This Implementation(LSTM-CRF-PA}|H + A|62.83(P) 65.47(R) 64.12(F1)|60.70(P) 62.75(R) 61.70(F1)
LSTM-CRF-PA+SL|H + A|62.31(P) 63.79(R) 63.04(F1)|61.57(P) 61.33(R) 61.45(F1)
This Implementation(LSTM-CRF-PA+SL)|H + A|64.29(P) 66.32(R) 65.28(F1)|59.33(P) 61.33(R) 60.31(F1)

### Dataset: MSRA
Model|Training Data|Dev|Test
-|:-:|:-:|:-:
LSTM-CRF|H|85.21(P) 78.91(R) 81.94(F1)|78.50(P) 74.50(R) 76.45(F1)
This Implementation(LSTM-CRF)|H|89.72(P) 79.17(R) 84.11(F1)|85.78(P) 73.90(R) 79.40(F1)
LSTM-CRF|H + A|87.00(P) 65.20(R) 74.54(F1)|83.41(P) 58.96(R) 69.09(F1)
This Implementation(LSTM-CRF)|H + A|86.70(P) 66.46(R) 75.24(F1)|84.34(P) 62.75(R) 71.76(F1)
LSTM-CRF-PA|H + A|83.78(P) 81.79(R) 82.77(F1)|79.19(P) 77.59(R) 78.38(F1)
This Implementation(LSTM-CRF-PA}|H + A|86.09(P) 82.89(R) 84.46(F1)|82.27(P) 79.48(R) 80.85(F1)
LSTM-CRF-PA+SL|H + A|86.94(P) 80.12(R) 83.40(F1)|81.63(P) 76.95(R) 79.22(F1)
This Implementation(LSTM-CRF-PA+SL)|H + A|89.99(P) 82.37(R) 86.01(F1)|84.78(P) 77.69(R) 81.08(F1)
