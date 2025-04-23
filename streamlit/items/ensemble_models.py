import streamlit as st

def ensemble_models():

    st.write('<h1>Ensemble Models</h1>', unsafe_allow_html = True)

    st.write("""We created a custom model that extracted the final feature
             layer from the text and image models and then fused the features
             using torch.concat([text, image], dim = 1). The model features
             were then passed through a more complex classifier that had
             three linear layers, accompanied by batch normalization and
             two dropout layers to prevent overfitting and enhance
             classification. The models we trained were derived from a
             combination of the models we trained previously: <br>
             DenseNet169 + roBERTa<br>
             ResNet152 + roBERTa<br>
             DenseNet169 + roBERTa Mulitlanguage<br>
             ResNet152 + roBERTa Mulitlanguage<br>
             During training, we found that because the text score was so
             high already, the model did quickly begin to memorize the
             training data, with lower increases in validation scores.
             Typically, early stopping typically occured before before ten
             epochs. In conjunction with experimenting with learning rate,
             beta parameters, and weight decay, we found the best model
             was DenseNet169 + roBERTa Multilanguage. We suspect the
             DenseNet169 feature map complemented roBERTa's much better
             during feature fusion than ResNet152, even though ResNet152
             was able to achieve a better score on images alone than
             DenseNet169. Please see the model comparisons for final F1-Scores
             of all models, including the multimodal models. Below, we
             present the Training, Validation, and Test scores for the best
             model.""",
             unsafe_allow_html = True)

    st.image('images/roBERat_multi+densenet_f1_scores.png')
