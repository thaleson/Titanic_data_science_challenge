import streamlit as st
from streamlit_option_menu import option_menu




# Configuração da página
st.set_page_config(
    page_title='Desafio do Titanic 🚢💔',
    page_icon='🚢',
    layout='wide'
)

# Aplicar estilos de CSS à página
with open("static/style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Definir o menu de navegação com ícones
with st.sidebar:
    st.title("Menu")
    option = option_menu(
        menu_title=None,  # Título do menu (opcional)
        options=["Home", "Sobre o Projeto", "Previsão Titanic"],
        icons=["house", "info-circle", "cash"],  # Ícones para cada opção
        menu_icon="cast",  # Ícone do menu
        default_index=0,  # Índice da opção selecionada por padrão
        orientation="vertical"  # Orientação do menu (vertical ou horizontal)
    )

    st.markdown("""
        <p style="text-align: center;">Meus contatos</p>
        """, unsafe_allow_html=True)

    # Badges
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between;">
            <div>
                <a href="https://github.com/thaleson" target="_blank">
                    <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" width="100" />
                </a>
            </div>
            <div>
                <a href="https://www.linkedin.com/in/thaleson-silva-9298a0296/" target="_blank">
                    <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" width="100" />
                </a>
            </div>
            <div>
                <a href="mailto:thaleson177@gmail.com" target="_blank">
                    <img src="https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white" width="80" />
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Carregar a página selecionada
if option == "Home":
    from pages.nav.home import show_home
    show_home()
elif option == "Sobre o Projeto":
    from pages.nav.about import show_about
    show_about()
elif option == "Previsão Titanic":
    from pages.nav.predict_survival import main  # Ajuste aqui
    main()  # Chame a função principal da página de previsão
