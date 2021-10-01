# DSNER
Pytorch implementation to paper "Distantly Supervised NER with Partial Annotation Learning and Reinforcement Learning".

# Train
see main.py

# Cite
If you use the code or data, please cite the following paper:

[Yang et al., 2018] Yaosheng Yang, Wenliang Chen, Zhenghua Li, Zhengqiu He and Min Zhang. Distantly Supervised NER with Partial Annotation Learning and Reinforcement Learning, Proceedings of COLING2018, pp.2159–2169, Santa Fe, New Mexico, USA, August 20-26, 2018

# Performance Comparision

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H|EC-Dev|63.78|**61.26**|**62.49**
This Implementation(LSTM-CRF)|H|EC-Dev|**65.14**|59.79|62.35

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H|EC-Test|59.93|**58.46**|59.19
This Implementation(LSTM-CRF)|H|EC-Test|**62.81**|57.41|**59.99**


Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H + A|EC-Dev|67.75|52.91|59.42
This Implementation(LSTM-CRF)|H + A|EC-Dev|**69.27**|**54.11**|**60.76**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H + A|EC-Test|62.36|48.54|54.59
This Implementation(LSTM-CRF)|H + A|EC-Test|**65.77**|**50.44**|**57.09**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF-PA|H + A|EC-Dev|60.34|64.49|62.35
This Implementation(LSTM-CRF-PA}|H + A|EC-Dev|**62.83**|**65.47**|**64.12**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF-PA|H + A|EC-Test|59.36|60.82|60.08
This Implementation(LSTM-CRF-PA}|H + A|EC-Test|**60.70**|**62.75**|**61.70**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF-PA+SL|H + A|EC-Dev|62.31|63.79|63.04
This Implementation(LSTM-CRF-PA+SL)|H + A|EC-Dev|**64.29**|**66.32**|**65.28**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF-PA+SL|H + A|EC-Test|**61.57**|61.33|**61.45**
This Implementation(LSTM-CRF-PA+SL)|H + A|EC-Test|59.33|61.33|60.31

---

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H|NEWS-Dev|85.21|78.91|81.94
This Implementation(LSTM-CRF)|H|NEWS-Dev|**89.72**|**79.17**|**84.11**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H|NEWS-Test|78.50|**74.50**|76.45
This Implementation(LSTM-CRF)|H|NEWS-Test|**85.78**|73.90|**79.40**


Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H + A|NEWS-Dev|**87.00**|65.20|74.54
This Implementation(LSTM-CRF)|H + A|NEWS-Dev|86.70|**66.46**|**75.24**


Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF|H + A|NEWS-Test|83.41|58.96|69.09
This Implementation(LSTM-CRF)|H + A|NEWS-Test|**84.34**|**62.75**|**71.76**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF-PA|H + A|NEWS-Dev|83.78|81.79|82.77
This Implementation(LSTM-CRF-PA}|H + A|NEWS-Dev|**86.09**|**82.89**|**84.46**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF-PA|H + A|NEWS-Test|79.19|77.59|78.38
This Implementation(LSTM-CRF-PA}|H + A|NEWS-Test|**82.27**|**79.48**|**80.85**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF-PA+SL|H + A|NEWS-Dev|86.94|80.12|83.40
This Implementation(LSTM-CRF-PA+SL)|H + A|NEWS-Dev|**89.99**|**82.37**|**86.01**

Model|Training Data|Dataset|Precision|Recall|F1
:-:|:-:|:-:|:-:|:-:|:-:
LSTM-CRF-PA+SL|H + A|NEWS-Test|81.63|76.95|79.22
This Implementation(LSTM-CRF-PA+SL)|H + A|NEWS-Test|**84.78**|**77.69**|**81.08**
