import streamlit as st
from items.project_description import project_description
from items.data_description import data_description
from items.text_models import text_models
from items.image_models import image_models
from items.ensemble_models import ensemble_models
from items.compare_conclude import compare_conclude

st.html(
    """
<style>
[data-testid="stSidebarContent"] {
    background: white;
    /* Gradient background */
    color: white; /* Text color */
    padding: 5px; /* Add padding */
}

/* Main content area */
[data-testid="stAppViewContainer"] {
    background: white;
    padding: 5px; /* Add padding for the main content */
    border-radius: 5px; /* Add rounded corners */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* Add a subtle shadow */
}

/* Apply Times New Roman font style globally */
body {
    font-family: 'Roboto', sans-serif;
    font-size: 16px; /* Set the font size */
    color: black; /* Set text color */   
}

/* style other elements globally */
h1, h2, h3 {
    font-family: 'Roboto', sans-serif;
    color: black; /* Set a color for headers */
    width: 100% !important;
}

/* Customize the sidebar text */
[data-testid="stSidebarContent"] {
    font-family: 'Roboto', sans-serif;
    color: black;
}

/* Change the text color of the entire sidebar */
[data-testid="stSidebar"] {
    color: black !important;
}

/* Change the color of the radio button labels */
.stRadio label {
    color: black !important;
}

/* Change the color of the radio button option text */
.stRadio div {
    color: black !important;
}

/* Change the text color for the entire main content area */
body {
    color: black !important;
}

/* Change the color of text in markdown and other text elements */
.stMarkdown, .stText {
    color: black !important;
}

/* Adjust the width of the main content area */
div.main > div {
    width: 80% !important;
    margin: 0 auto;  /* Center the content */
}

</style>
"""
)

st.sidebar.image("images/rakuten_logo.png", use_container_width = False)

menu = st.sidebar.radio("Menu", ["Poject Description",
                                 "Data Description",
                                 "Text Models",
                                 "Image Models",
                                 "Ensemble Models",
                                 "Model Comparisons & Conclusions"],
                        label_visibility = "collapsed")

if menu == "Poject Description":
    project_description()
    
elif menu == "Data Description":
    data_description()
    
elif menu == "Text Models":
    text_models()

elif menu == "Image Models":
    image_models()

elif menu == "Ensemble Models":
    ensemble_models()

elif menu == "Model Comparisons & Conclusions":
    compare_conclude()

st.sidebar.header('Collaborators')

st.sidebar.write("""<b>Phil J. Howson</b><br>Project Lead, Image Processing & Modelling,
                    Mulitlanguage Models, Ensemble Models""", unsafe_allow_html = True)

col1, col2 = st.sidebar.columns([0.15, 0.85], gap = "small",
                                vertical_alignment = "center")
col1.image("images/github.png")
col2.write("""<a href = "https://github.com/philjhowson">GitHub</a>""",
           unsafe_allow_html = True)

col1, col2 = st.sidebar.columns([0.15, 0.85], gap = "small",
                                vertical_alignment = "center")
col1.image("images/linkedin.png")
col2.write("""<a href = "https://www.linkedin.com/in/philjhowson/">LinkedIn</a>""",
           unsafe_allow_html = True)

st.sidebar.write("""<b>Torben Hänke</b><br>Text Processing & Modelling""", unsafe_allow_html = True)

col1, col2 = st.sidebar.columns([0.15, 0.85], gap = "small",
                                vertical_alignment = "center")
col1.image("images/github.png")
col2.write("""<a href = "https://github.com/torben-haenke">GitHub</a>""",
           unsafe_allow_html = True)

st.sidebar.write("""<b>Hui Zhang</b><br>Text Processing & Modelling""", unsafe_allow_html = True)

col1, col2 = st.sidebar.columns([0.15, 0.85], gap = "small",
                                vertical_alignment = "center")
col1.image("images/github.png")
col2.write("""<a href = "https://github.com/HuiZhang95">GitHub</a>""",
           unsafe_allow_html = True)


