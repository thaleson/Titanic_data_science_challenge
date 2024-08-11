import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def show_about():
    # Aplicando estilos personalizados usando CSS
    st.markdown("""
        <style>
            h1, h2, h3 {
                color: blue;
            }
            p {
                font-size: 16px;
                line-height: 1.6;
                color: #ffffff;
                text-align: justify;
            }
            .stButton > button {
                background-color: #4CAF50;
                color: white;
            }
            .stButton > button:hover {
                background-color: #45a049;
            }
        </style>
    """, unsafe_allow_html=True)

    # Carregando o dataset
    train_df = pd.read_csv('data/train.csv')

    st.title("Sobre o Projeto")
    st.header("Análise Exploratória e Modelagem")

    # 1. Introdução ao Projeto
    st.write("""
    ### Introdução:
    🌟 O objetivo deste projeto é prever se um passageiro sobreviveria ao desastre do Titanic, com base em características como classe social, sexo, idade, número de irmãos/cônjuges a bordo, número de pais/filhos a bordo, tarifa do bilhete e porto de embarque.

    Utilizamos um modelo de aprendizado de máquina para realizar essas previsões, e a seguir, detalhamos o processo de análise exploratória e construção do modelo.

    📈 **Data Science**: Este projeto envolve análise exploratória dos dados e modelagem preditiva para entender melhor os fatores que influenciam a sobrevivência.
    """)

    # Adicionando a imagem
    st.image('static/imgs/titanicimg.jpg', caption='Titanic', use_column_width=True)

    # 2. Passo a Passo da Análise
    st.write("""
    ### Passo a Passo da Análise Exploratória e Construção do Modelo:

    1. **Carregamento dos Dados**:
    ```python
    import pandas as pd
    train_df = pd.read_csv('data/train.csv')
    ```

    2. **Visualização das Primeiras Linhas**:
    ```python
    train_df.head()
    ```
    """)
    st.write("Vamos visualizar as primeiras linhas do dataset:")
    st.write(train_df.head())

    # 3. Distribuição da Idade dos Passageiros
    st.write("""
    3. **Distribuição da Idade dos Passageiros**:
    ```python
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 5))
    sns.histplot(train_df['Age'].dropna(), kde=False, bins=30)
    plt.title('Distribuição da Idade dos Passageiros')
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.show()
    ```
    """)
    st.write("**Gráfico: Distribuição da Idade dos Passageiros:**")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(train_df['Age'].dropna(), kde=False, bins=30, ax=ax)
    ax.set_title('Distribuição da Idade dos Passageiros')
    ax.set_xlabel('Idade')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

    # 4. Análise da Sobrevivência por Sexo
    st.write("""
    4. **Análise da Sobrevivência por Sexo**:
    ```python
    sns.countplot(x='Sex', hue='Survived', data=train_df)
    plt.title('Sobrevivência por Sexo')
    plt.xlabel('Sexo')
    plt.ylabel('Contagem')
    plt.show()
    ```
    """)
    st.write("**Gráfico: Sobrevivência por Sexo:**")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='Sex', hue='Survived', data=train_df, ax=ax)
    ax.set_title('Sobrevivência por Sexo')
    ax.set_xlabel('Sexo')
    ax.set_ylabel('Contagem')
    st.pyplot(fig)

    # 5. Análise da Sobrevivência por Classe
    st.write("""
    5. **Análise da Sobrevivência por Classe**:
    ```python
    sns.countplot(x='Pclass', hue='Survived', data=train_df)
    plt.title('Sobrevivência por Classe')
    plt.xlabel('Classe')
    plt.ylabel('Contagem')
    plt.show()
    ```
    """)
    st.write("**Gráfico: Sobrevivência por Classe:**")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='Pclass', hue='Survived', data=train_df, ax=ax)
    ax.set_title('Sobrevivência por Classe')
    ax.set_xlabel('Classe')
    ax.set_ylabel('Contagem')
    st.pyplot(fig)

    # 6. Modelo de Análise Utilizado
    st.write("""
    ### Modelo de Análise Utilizado:
    Utilizamos um modelo de **Regressão Logística** para prever a sobrevivência dos passageiros. A Regressão Logística é um método estatístico que modela a probabilidade de um evento ocorrer, neste caso, a sobrevivência.

    O processo de construção do modelo incluiu:

    1. **Pré-processamento dos Dados**: Tratamento de valores ausentes, codificação de variáveis categóricas e normalização de dados numéricos.
    2. **Divisão do Dataset**: Separação dos dados em conjuntos de treino e teste para avaliação da performance do modelo.
    3. **Treinamento do Modelo**: Ajuste do modelo de Regressão Logística aos dados de treino.
    4. **Avaliação do Modelo**: Medição da acurácia e análise da matriz de confusão para avaliar a capacidade preditiva do modelo.

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix

    # Pré-processamento dos Dados
    X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = train_df['Survived']

    # Remoção de linhas com valores ausentes
    X = X.dropna()
    y = y[X.index]

    # Codificação das variáveis categóricas
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    X['Embarked'] = X['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Divisão do Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinamento do Modelo
    model = LogisticRegression()
    model.fit(X_train, y_train)
    ```

    **Avaliação do Modelo**:
    ```python
    # Avaliação do Modelo
    y_pred = model.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    matriz_confusao = confusion_matrix(y_test, y_pred)

    # Exibição da Acurácia
    acuracia

    # Exibição da Matriz de Confusão
    matriz_confusao
    ```
    """)

    # Calcular e mostrar a acurácia e matriz de confusão
    X = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = train_df['Survived']

    # Remoção de linhas com valores ausentes
    X = X.dropna()
    y = y[X.index]

    # Codificação das variáveis categóricas
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
    X['Embarked'] = X['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

    # Divisão do Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinamento do Modelo
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    matriz_confusao = confusion_matrix(y_test, y_pred)

    # Exibição da Acurácia
    st.write("**Acurácia do Modelo:**")
    st.write(f"Acurácia: {acuracia:.2f}")

    # Exibição da Matriz de Confusão
    st.write("**Matriz de Confusão:**")
    st.write(matriz_confusao)

    # 7. Conclusão
    st.write("""
    ### Conclusão:
    Este projeto demonstra como um modelo de aprendizado de máquina pode ser aplicado para prever a sobrevivência em cenários complexos, como o desastre do Titanic. Através de uma análise detalhada dos dados e da construção de um modelo preditivo, conseguimos obter uma acurácia razoável, indicando que variáveis como classe social, sexo e idade tiveram um impacto significativo na probabilidade de sobrevivência.



    🔗 Confira o código completo no meu GitHub: [Thaleson Silva](https://github.com/thaleson)
    """)