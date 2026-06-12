import numpy as np
import matplotlib.pyplot as plt

# PARTE 3 - IMPLEMENTAÇÃO DAS FUNÇÕES

# 3.1 Funções

def fun_sigmoide_binaria(x):
  return 1.0 / (1.0 + np.exp(-x))

def fun_sigmoide_bipolar(x):
  return (2 / (1 + np.exp(-x))) - 1

def tan_hiperbolica(x):
  return np.tanh(x)

def derivada(x):
  return x * (1.0 -x)