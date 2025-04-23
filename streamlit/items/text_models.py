import streamlit as st

def text_models():

    st.write('<h1>Text Models</h1>', unsafe_allow_html = True)

    st.write('<h2>roBERTa</h2>', unsafe_allow_html = True)

    st.write("""Our first model was roBERTa-base, trained only on English
             data. We used the translated text and implemented a finally
             formatting of the data using spacy. We tokenized, lemmatized,
             removed stopwords and punctuation, and any unnecessary
             white space, and finally converted words to lower case. We
             then began model training. The tokenizer had a length of
             128. We used a batch size of 64 due to hardware limitations.
             We froze all the layers of the roBERTa model, except the final
             5 layers. We used an Adam optimizer and we set the starting
             learning rate at 1e-4, due to benefits of a slower rate when
             training deeper layers. However, we did also experiment with
             1e-3 and 1e-5 starting rates. We used a learning rate scheduler
             with a patience of 3 epochs and a reduction factor of 0.1. We
             also computed class weights using scikit-learn's
             compute_class_weights function, with the 'balanced' class_weight
             setting. We monitored training and validation F1-Score and
             Loss and the training gradients. We utilized an early stopping
             function that would restore the model with the best F1-Score if
             no reduction in loss was seen after 6 epochs or if no reduction
             in F1-Score was seen in 10 epochs. Early stopping triggered at
             9 epoch due to a gradually increasing loss. Even though there
             was a slight increase in F1-Score over time, this stop was
             necessary to prevent severe overfitting. The model found
             meaningful connections between target words. For example,
             for Class 60, the model found a meaningful connection with
             words such as 'Playstation' and 'Sony' and class membership.
             Below we present an image of the training history and an
             example of feature imporance extraction for a single class.""")

    st.image('images/roBERTa_training_history.png')

    st.image('images/mean_attention_map_roBERTa_class_60.png')

    st.write('<h2>roBERTa multilanguage</h2>', unsafe_allow_html = True)

    st.write("""Training of the roBERTa multilanguage model was similar to
             roBERTa. However, we had to import multiple languages from spacy
             in order to appropriately pre-processed text. The final configurations
             ended up being the same as with roBERTa. However, early stopping was
             triggered later, but also due to a gradually increasing validation
             loss. roBRTa multilanguage found similar connections as roBERTa
             with words such as 'Playstation' and 'Sony' for Class 60, but
             the intensity of the connections were different. Below we present
             an image of the training history and an example of feature
             imporance extraction for a single class.""")

    st.image('images/roBERTa_multi_training_history.png')

    st.image('images/mean_attention_map_roBERTa_multi_class_60.png')

    st.write('<h3>Discussion</h3>', unsafe_allow_html = True)

    st.write("""We found that both roBERTa and roBERTa multilanguage perform
             well, given the compexity of the task. However, we did find slightly
             better performance (i.e., lower loss, higher F1-Score, and less
             overfitting) for roBERTa multilanguage. We suspect that some loss
             in information was due to translation inconsistencies. Despite this
             both models performed better on the validation set than the baseline
             provided by Rakuten.""")

    
