import streamlit as st
import pandas as pd
import joblib

# Carregar o modelo treinado com cache
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/titanic_survival_model.pkl')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

# Função para prever a sobrevivência
def predict_survival(model, perfil):
    try:
        if model is None:
            raise ValueError("O modelo não está carregado corretamente.")
        
        perfil_df = pd.DataFrame([perfil])
        probabilidade_sobreviver = model.predict_proba(perfil_df)[0][1]
        return probabilidade_sobreviver
    except Exception as e:
        st.error(f"Erro na previsão de sobrevivência: {e}")
        return None

# Função principal do Streamlit
def main():
    model = load_model()
    
    if model is None:
        st.error("Não foi possível carregar o modelo. Verifique o arquivo e tente novamente.")
        return

    st.title("Você sobreviveria ao Titanic 🚢💔?")

    # Entrada dos dados do usuário
    st.header("Insira suas características:")
    try:
        pclass = st.selectbox('Classe', [1, 2, 3], index=0)
        sex = st.selectbox('Sexo', ['Masculino', 'Feminino'], index=0)
        age = st.slider('Idade', 0, 100, 0)
        sibsp = st.number_input('Número de irmãos/cônjuges a bordo', min_value=0, max_value=10, value=0)
        parch = st.number_input('Número de pais/filhos a bordo', min_value=0, max_value=10, value=0)
        fare = st.number_input('Tarifa do bilhete', min_value=0.0, max_value=500.0, value=0.0)
        embarked = st.selectbox('Porto de Embarque', ['Cherbourg', 'Queenstown', 'Southampton'], index=2)

        # Mapeamento das entradas
        sex = 0 if sex == 'Masculino' else 1
        embarked = {'Cherbourg': 0, 'Queenstown': 1, 'Southampton': 2}[embarked]

        perfil = {
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked
        }

        # Verifica se o botão foi clicado
        if st.button("Prever Sobrevivência"):
            try:
                probabilidade = predict_survival(model, perfil) * 100
                if probabilidade is not None:
                    # Formatar a probabilidade com duas casas decimais
                    probabilidade_formatada = f"{probabilidade:.2f}"
                    if probabilidade < 50:
                        st.error(f"Uma pena, você não sobreviveria. Probabilidade de sobrevivência: {probabilidade_formatada}%")
                    else:
                        st.success(f"Você sobreviveria! Probabilidade de sobrevivência: {probabilidade_formatada}%")
                else:
                    st.error("Não foi possível calcular a probabilidade de sobrevivência.")
            except Exception as e:
                st.error(f"Erro ao fazer a previsão: {e}")
    except Exception as e:
        st.error(f"Erro ao coletar entradas do usuário: {e}")

if __name__ == "__main__":
    main()
