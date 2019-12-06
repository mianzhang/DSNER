# Performance Comparision

Model|Training Data|Dev|Test
-|:-:|:-:|:-:
LSTM-CRF|H|63.78(P) 61.26(R) 62.49(F1)|59.93(P) 58.46(R) 59.19(F1)
This Implementation(LSTM-CRF)|H|63.76(P) 60.00(R) 61.82(F1)|64.19(P) 60.13(R) 62.09(F1)
LSTM-CRF|H + A|67.75(P) 52.91(R) 59.42(F1)|62.36(P) 48.54(R) 54.59(F1)
This Implementation(LSTM-CRF)|H + A|68.35(P) 51,37(R) 58.66(F1)|67.49(P) 50.44(R) 57.73(F1)
LSTM-CRF-PA|H + A|60.34(P) 64.49(R) 62.35(F1)|59.36(P) 60.82(R) 60.08(F1)
This Implementation(LSTM-CRF-PA}|H + A|61.90(P) 64.63(R) 63.23(F1)|61.90(P) 63.73(R) 62.80(F1)
LSTM-CRF-PA+SL|H + A|62.31(P) 63.79(R) 63.04(F1)|61.57(P) 61.33(R) 61.45(F1)
This Implementation(LSTM-CRF-PA+SL)|H + A|62.02(P) 64.63(R) 63.30(F1)|60.86(P) 62.85(R) 61.84(F1)
