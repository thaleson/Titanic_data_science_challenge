# 🚢 Titanic Challenge: Previsão de Sobrevivência

Bem-vindo ao projeto **Titanic Challenge**! Este projeto visa prever a sobrevivência dos passageiros do Titanic com base em características como classe social, sexo, idade, número de irmãos/cônjuges a bordo, número de pais/filhos a bordo, tarifa do bilhete e porto de embarque.

## 📊 Sobre o Projeto

O objetivo deste projeto é aplicar técnicas de aprendizado de máquina para prever se um passageiro sobreviveria ao desastre do Titanic. Utilizamos um modelo de Regressão Logística para realizar essas previsões e apresentamos a análise exploratória e a construção do modelo em uma aplicação interativa com Streamlit.

### 🔍 Análise Exploratória

A análise inclui:
- Visualização da distribuição da idade dos passageiros.
- Análise da sobrevivência por sexo e classe.
- Criação de gráficos para visualizar insights dos dados.

### 🔧 Modelagem

Utilizamos um modelo de **Regressão Logística** para prever a sobrevivência:
1. **Pré-processamento dos Dados**: Tratamento de valores ausentes, codificação de variáveis categóricas e normalização.
2. **Divisão do Dataset**: Separação dos dados em conjuntos de treino e teste.
3. **Treinamento do Modelo**: Ajuste do modelo aos dados de treino.
4. **Avaliação do Modelo**: Medição da acurácia e análise da matriz de confusão.

## 🛠️ Requisitos

Certifique-se de ter os seguintes pacotes instalados:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

## 🚀 Executando o Projeto

Para executar a aplicação Streamlit, use o seguinte comando:

```bash
streamlit run app.py
```

## 📁 Estrutura do Projeto

- **`app.py`**: Arquivo principal que inicia a aplicação Streamlit.
- **`pages/`**: Pasta contendo diferentes páginas da aplicação.
  - **`nav/`**: Pasta com arquivos relacionados à navegação e ao conteúdo.
- **`data/`**: Pasta com o dataset do Titanic.
- **`requirements.txt`**: Lista de dependências do projeto.
- **`.gitignore`**: Arquivo para excluir arquivos desnecessários do repositório.

## ✨ Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests para melhorias.

## 📜 Licença

Este projeto é licenciado sob a MIT License - veja o [LICENSE](LICENSE) para mais detalhes.

## 🙌 Agradecimentos

Agradecemos a todos que contribuíram para este projeto e todos que o utilizam e testam. Esperamos que você encontre este projeto útil e informativo!

---

**Obrigado por explorar o Titanic Challenge!** 🚢🔍
