import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import redeNeural  

# Configuração da página do Streamlit
st.set_page_config(page_title="Experimentos XOR - MLP", layout="wide")

st.title("🧠 Simulador de Rede Neural MLP - Experimento XOR")
st.markdown("Interface interativa para as Partes 6, 7 e 8 do projeto de Redes Neurais.")

# --- BARRA LATERAL DE CONTROLES ---
st.sidebar.header("🎛️ Configurações da Rede")

# Seleção da Função Lógica (Desafio Opcional incluso!)
funcao_escolhida = st.sidebar.selectbox(
    "Selecione a Função Lógica:", 
    ["XOR", "AND", "OR", "NAND", "NOR"]
)

# Experimento A: Quantidade de neurônios ocultos
qtd_ocultos = st.sidebar.slider("Neurônios na Camada Oculta (Experimento A):", min_value=2, max_value=5, value=4)

# Experimento B: Taxa de aprendizagem
tx_aprendizado = st.sidebar.selectbox("Taxa de Aprendizagem (Experimento B):", [0.1, 0.2, 0.3, 0.4, 0.5], index=1)

# Experimento C: Influência da Inicialização (Semente Aleatória)
semente = st.sidebar.number_input("Semente Aleatória (Experimento C):", min_value=0, max_value=100, value=42)

# Configurações de Parada
tolerancia = st.sidebar.number_input("Tolerância do Erro (Critério de Parada):", value=0.001, format="%.4f")
max_epocas = st.sidebar.number_input("Máximo de Épocas:", value=20000, step=5000)

# Matriz global de entradas utilizada na renderização posterior
entradas_globais = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


# --- FUNÇÃO DE TREINAMENTO PROTEGIDA POR CACHE (ACELERA O SITE) ---
@st.cache_data
def rodar_treinamento_com_cache(func_escolhida, num_ocultos, taxa, sem, tol, epocas_limite):
    # Forçamos a semente para garantir reprodutibilidade nos experimentos
    np.random.seed(sem)
    
    # Criamos os dados locais para a função interna
    entradas_locais = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    
    if func_escolhida == "XOR": saidas_locais = np.array([0, 1, 1, 0])
    elif func_escolhida == "AND": saidas_locais = np.array([0, 0, 0, 1])
    elif func_escolhida == "OR": saidas_locais = np.array([0, 1, 1, 1])
    elif func_escolhida == "NAND": saidas_locais = np.array([1, 1, 1, 0])
    elif func_escolhida == "NOR": saidas_locais = np.array([1, 0, 0, 0])

    # Inicializa a rede neural MLP dinamicamente
    rede_local = redeNeural.RedeNeuralMLP(tx_aprendizado=taxa, qtd_ocultos=num_ocultos)
    
    # Treina coletando os históricos solicitados no relatório
    historico, ep_concluidas, err_final = rede_local.treinar(
        entradas_locais, saidas_locais, epocas=epocas_limite, tolerancia=tol
    )
    
    # Retorna o modelo treinado junto com os dados calculados de progresso
    return historico, ep_concluidas, err_final, rede_local


# --- ACIONAMENTO DO FLUXO DE EXECUÇÃO ---
if st.sidebar.button("🚀 Iniciar Treinamento", use_container_width=True):
    with st.spinner("Treinando a rede neural... Por favor, aguarde."):
        # Chamada da função protegida por cache do Streamlit
        historico_erros, epocas_concluidas, erro_final, rede = rodar_treinamento_com_cache(
            funcao_escolhida, qtd_ocultos, tx_aprendizado, semente, tolerancia, max_epocas
        )
        
    st.success(f"⚡ Treinamento concluído em {epocas_concluidas} épocas!")
    
    # Mapeando em duas colunas organizadas na interface web
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Gráfico: Erro x Época")
        # Exibe o gráfico de linhas interativo nativo do Streamlit
        st.line_chart(historico_erros, x_label="Época", y_label="Erro Quadrático Médio (MSE)")
        
        # Exibe os KPIs numéricos requeridos no experimento
        st.metric(label="Épocas até a Convergência", value=epocas_concluidas)
        st.metric(label="Erro Absoluto Final (MSE)", value=f"{erro_final:.6f}")

    with col2:
        st.subheader(f"📋 Saídas da Rede Após Treinamento ({funcao_escolhida})")
        
        # Redefine a saída esperada localmente para preencher a planilha visual
        if funcao_escolhida == "XOR": saidas_esperadas = np.array([0, 1, 1, 0])
        elif funcao_escolhida == "AND": saidas_esperadas = np.array([0, 0, 0, 1])
        elif funcao_escolhida == "OR": saidas_esperadas = np.array([0, 1, 1, 1])
        elif funcao_escolhida == "NAND": saidas_esperadas = np.array([1, 1, 1, 0])
        elif funcao_escolhida == "NOR": saidas_esperadas = np.array([1, 0, 0, 0])
        
        # Calcular as predições finais para montar a tabela verdade
        resultados_tabela = []
        for x, y_esp in zip(entradas_globais, saidas_esperadas):
            _, res_final = rede.forwardPass(x)
            
            # SOLUÇÃO REAL: .item() extrai o valor numérico puro de dentro do array do NumPy sem falhas
            valor_saida = float(res_final.item())
            classificacao = 1 if valor_saida >= 0.5 else 0
            
            resultados_tabela.append({
                "Entrada (X1, X2)": str(x),
                "Saída Esperada (Y)": int(y_esp),
                "Saída Bruta (Sigmóide)": f"{valor_saida:.4f}",
                "Classificação Final": classificacao
            })
            
        # Exibe os dados em formato de planilha interativa na tela
        st.dataframe(resultados_tabela, use_container_width=True)
        
        # --- EXIBIÇÃO DA FRONTEIRA DE DECISÃO ---
        st.subheader("🗺️ Mapa da Fronteira de Decisão")
        x_min, x_max = -0.5, 1.5
        y_min, y_max = -0.5, 1.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        pontos_grade = np.c_[xx.ravel(), yy.ravel()]
        
        previsoes = []
        for ponto in pontos_grade:
            _, resultado = rede.forwardPass(ponto)
            # SOLUÇÃO APLICADA AQUI TAMBÉM: .item() garante o valor puro para o mapa do gráfico
            previsoes.append(1 if float(resultado.item()) >= 0.5 else 0)
            
        Z = np.array(previsoes).reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(5, 4))
        # Define as cores do plano de fundo
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        
        # Plota os marcadores sobre o mapa
        for entrada, esperado in zip(entradas_globais, saidas_esperadas):
            cor = 'red' if esperado == 1 else "blue"
            marcador = 'X' if esperado == 1 else 'o'
            ax.scatter(entrada[0], entrada[1], color=cor, marker=marcador, s=100, edgecolors='k')
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig)
else:
    st.info("💡 Configure os parâmetros na barra lateral esquerda e clique em 'Iniciar Treinamento' para rodar os testes.")
