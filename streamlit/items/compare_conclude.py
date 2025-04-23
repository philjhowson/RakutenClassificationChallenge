import streamlit as st

def compare_conclude():

    st.write('<h1>Model Comparisons</h1>', unsafe_allow_html = True)

    st.write("""We compared the test F1-Scores for each other models
             to assess the quality of each model on unseen data. We
             found that the best performing model was DenseNet169 +
             roBERTa multilanguage, with a test F1-Score of 0.883. This
             was only below the validation score  by 0.003, suggesting
             consistency in the generalization to unseen data.""")

    st.image('images/model_f1_comparisons.png')

    st.write('<h3>Conclusions</h3>', unsafe_allow_html = True)

    st.write("""Fusion between image and text features have a high
             potential to increase model strength and generalizability,
             with a cost of being larger and more time consuming to train
             than simple image or text models. Their behaviour also appears
             to be similar to bagging or boosting algorithms in that a
             combination of the strongest models do not seem to guarantee
             the best score. Rather, a combination of lower performing models
             that complement each other more effectively in their accuracy
             produce much stronger results overall. The combination of high
             performing models could achieve better scores in-so-far as they
             compliment each others weaknesses. In the case we present here,
             DenseNet169 underperformed ResNet152 by a considerable margin on
             images alone, but it combined well with roBERTa Multilingual and
             complemented the weakness. In order to better enhance the output
             of the models, further methods in data cleaning, or text augmentation
             similar to the idea of how image augmentation is done with
             transforms.Compose() could further enhance model generalizability.""")
