import streamlit as st
import json
from streamlit_lottie import st_lottie

def show_home():
    # Título principal
    st.title("Desafio do Titanic 🚢💔")

    # Subtítulo
    st.subheader("Olá! Eu sou Thaleson Silva 👋")

    # Colunas que organizam a página
    col1, col2 = st.columns(2)

    # Animações
    with open("anims/animationship.json") as source:
        animacao_1 = json.load(source)

    with open("anims/Animation.json") as source:
        animacao_2 = json.load(source)

    # Conteúdo a ser exibido na coluna 1
    with col1:
        st_lottie(animacao_1, height=650, width=350)
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("""
            <h5 style='text-align: justify; line-height: 1.6;'>
                O projeto do Desafio do Titanic visa prever a sobrevivência dos passageiros do Titanic com base em suas características pessoais e sociais. Utilizando técnicas de aprendizado de máquina, nosso objetivo é analisar os dados históricos e construir um modelo preditivo para identificar a probabilidade de sobrevivência dos passageiros.
            </h5>
        """, unsafe_allow_html=True)

    # Conteúdo a ser exibido na coluna 2
    with col2:
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.write("")
        st.markdown("""
            <h5 style='text-align: justify; line-height: 1.6;'>
                Bem-vindo ao Desafio do Titanic! 🚢🔍
                Neste projeto, você pode explorar e analisar os dados do Titanic, visualizar insights e prever a sobrevivência dos passageiros. Utilizamos técnicas avançadas de aprendizado de máquina para fornecer uma análise precisa e interativa. Aproveite a oportunidade para descobrir como características como classe, sexo e idade influenciaram a sobrevivência no desastre do Titanic.
            </h5>
        """, unsafe_allow_html=True)
        st_lottie(animacao_2, height=400, width=440)
