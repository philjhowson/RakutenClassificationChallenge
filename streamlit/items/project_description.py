import streamlit as st

def project_description():

    st.image("images/rakuten_logo.png", use_container_width = True)
    
    st.write("<h1>Project Description</h1>", unsafe_allow_html = True)

    st.write("""Rakuten was created as an e-commerce plateform in Japan 
             in 1997. It is one of the largest platforms with over 1.3 
             billion members. Rakuten additionally makes strategic investments 
             in cutting edge technologies, which gave birth to the 
             Rakuten Institute of Techonology (RIT). RIT is a research 
             department that focuses on domains like computer vision, 
             natural language processing, machine & deep learning, and 
             human-computer interaction.""")

    st.write("""This challenge was issued by Rakuten to address large-scale 
             product type code multimodel classification. The goal is to 
             predict a product's type code based on image and text data. 
             The applications range from personalized search and 
             recommendations, to query understanding.""")

    st.write("""The challenge uses the weighted-F1 score due to class 
             imbalances in the training and test sets.""")

    st.write("<h2>Benchmark Model</h2>", unsafe_allow_html = True)

    st.write("""For text data, a simplified classifier was used. 
             The benchmark data only made use of the designation 
             column from the data. For the image classification, 
             ResNet50 was used with implementation in Keras. They unfroze 
             27 different layers from the top, including 8 convolutional 
             layers. The final network contained 12,144,667 training and 
             23,643,035 non-trainable parameters.""")

    st.write("<h2>Benchmark Performance</h2>", unsafe_allow_html = True)

    st.write("Text Model: 0.8113")
    st.write("Image Model: 0.5534")

    link = '<a href = "https://challengedata.ens.fr/challenges/35" target ="_blank">Challenge Homepage</a>'


    st.write("For full details about the challenge, the models used, "
             "top scores, and the dataset, visit the challenge webpage, "
             "linked below.")
             
    st.markdown(link, unsafe_allow_html = True)
