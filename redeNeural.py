
import numpy as np
import funcoes as funcoes 



class RedeNeuralMLP:
  """Inicialização de parâmetros passíveis de alteraçãoa
  
  Inicializado com uma camada oculta e uma de saída utilizando backpropagation para aprender
  """
  def __init__(self, tx_aprendizado=0.001, qtd_ocultos=4):
    self.taxa_aprendizado = tx_aprendizado
    # Valor do pesos definidos aleatoriamente baseado na quantidade dinâmica de neurônios
    self.pesos_entradas_ocultas = np.random.rand(2, qtd_ocultos)
    self.pesos_ocultos_saida = np.random.rand(qtd_ocultos, 1)
    self.bias_oculto = np.zeros(qtd_ocultos)
    self.bias_saida = np.zeros(1)

  def forwardPass(self, entradas):
    """Forward Pass
    
    cálculo da ativação das entradas até a saída
    """
    soma_ponderada = (
      np.dot(entradas, self.pesos_entradas_ocultas) + self.bias_oculto
    )
    # Função de ativação sigmóide
    saida_oculta = funcoes.fun_sigmoide_binaria(soma_ponderada)

    # Camada de saída
    # Soma da saída combinada com o cálcula da saída camada oculta
    soma_saida = (
      np.dot(saida_oculta, self.pesos_ocultos_saida) + self.bias_saida
    )
    # Ativação final (0 ou 1)
    saida_final = funcoes.fun_sigmoide_binaria(soma_saida)

    # Retorna ambas ativações para calcular o gradiente
    return saida_oculta, saida_final

  def treinar(self, entradas_treino, saidas_esperadas, epocas=10000, tolerancia=0.001):
    """ Treinamento por Retropropagação
    
    Erro monitorado para interrupção por tolerância
    """
    historico_erros = []
      
    for epoca in range(epocas):
      erro_total_epoca = 0
      
      for x, y_esperado in zip(entradas_treino, saidas_esperadas):
        # FORWARD PASS
        saida_oculta, saida_final = self.forwardPass(x)

        # Cálculo de erro da saída
        erro_saida = float(np.ravel(y_esperado - saida_final)[0])
        # Acumula para erro quadrático
        erro_total_epoca += erro_saida ** 2
        
        # BACKWARD PASS
        # Multiplicação da derivada da saída pela sigmóide 
        gradiente_saida = erro_saida * funcoes.derivada(saida_final)

        # Erro da camada oculta retropropaga o gradiente da saída
        erro_oculto = gradiente_saida * self.pesos_ocultos_saida.T
        # Erro multiplicado pela derivada da ativação oculta
        gradiente_oculto = erro_oculto * funcoes.derivada(saida_oculta)

        # Ajuste de pesos
        self.pesos_ocultos_saida += (self.taxa_aprendizado * gradiente_saida * saida_oculta.reshape(-1, 1))
        self.bias_saida += self.taxa_aprendizado * gradiente_saida

        # gera matriz 2x4
        grad_oculto_plano = gradiente_oculto.ravel()
        self.pesos_entradas_ocultas += (self.taxa_aprendizado * np.outer(x, grad_oculto_plano))
        self.bias_oculto += self.taxa_aprendizado * grad_oculto_plano
      
      # Calcula o erro quadrático
      erro_medio = erro_total_epoca / len(entradas_treino)
      historico_erros.append(erro_medio)
      
      # Para a execução caso a média fique abaixo do limiar
      if erro_medio < tolerancia:
        print(f"Treinamento interrompido na época {epoca} por atingir a tolerância. Erro: {erro_medio:.6f}")
        return historico_erros, epoca, erro_medio

    return historico_erros, epocas, erro_medio


