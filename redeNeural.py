
import numpy as np
import funcoes as funcoes 

class RedeNeuralMLP:
  def __init__(self, tx_aprendizado=0.001, qtd_ocultos=4):
    self.taxa_aprendizado = tx_aprendizado
    self.pesos_entradas_ocultas = np.random.rand(2, qtd_ocultos)
    self.pesos_ocultos_saida = np.random.rand(qtd_ocultos, 1)
    self.bias_oculto = np.zeros(qtd_ocultos)
    self.bias_saida = np.zeros(1)

  def forwardPass(self, entradas):
    soma_ponderada = (
      np.dot(entradas, self.pesos_entradas_ocultas) + self.bias_oculto
    )
    saida_oculta = funcoes.fun_sigmoide_binaria(soma_ponderada)

    soma_saida = (
      np.dot(saida_oculta, self.pesos_ocultos_saida) + self.bias_saida
    )
    saida_final = funcoes.fun_sigmoide_binaria(soma_saida)

    return saida_oculta, saida_final

  def treinar(self, entradas_treino, saidas_esperadas, epocas=10000, tolerancia=0.001):
      historico_erros = []
      
      for epoca in range(epocas):
        erro_total_epoca = 0
        
        for x, y_esperado in zip(entradas_treino, saidas_esperadas):
          # FORWARD PASS
          saida_oculta, saida_final = self.forwardPass(x)

          # Camada saída
          erro_saida = float(np.ravel(y_esperado - saida_final)[0])
          erro_total_epoca += erro_saida ** 2
          
          gradiente_saida = erro_saida * funcoes.derivada(saida_final)

          # Camada oculta
          erro_oculto = gradiente_saida * self.pesos_ocultos_saida.T
          gradiente_oculto = erro_oculto * funcoes.derivada(saida_oculta)

          self.pesos_ocultos_saida += (self.taxa_aprendizado * gradiente_saida * saida_oculta.reshape(-1, 1))
          self.bias_saida += self.taxa_aprendizado * gradiente_saida

          grad_oculto_plano = gradiente_oculto.ravel()
          self.pesos_entradas_ocultas += (self.taxa_aprendizado * np.outer(x, grad_oculto_plano))
          self.bias_oculto += self.taxa_aprendizado * grad_oculto_plano
        
        erro_medio = erro_total_epoca / len(entradas_treino)
        
        historico_erros.append(erro_medio)
        
        if erro_medio < tolerancia:
          print(f"Treinamento interrompido na época {epoca} por atingir a tolerância. Erro: {erro_medio:.6f}")
          return historico_erros, epoca, erro_medio

      return historico_erros, epocas, erro_medio


