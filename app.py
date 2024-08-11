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
