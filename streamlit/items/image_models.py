import streamlit as st

def image_models():

    st.write('<h1>Image Models</h1>', unsafe_allow_html = True)

    st.write('<h2>DenseNet169</h2>', unsafe_allow_html = True)

    st.write("""In order to faciliate better model training, we used
             transforms.Compose(), with a random vertical and horizontal
             flip, a random affine (± 90 degrees with 0.1 translation),
             image normalization, and our custom resizing function. The
             validation set only had resizing and normalization. We
             were able to utilize a batch size of 512 for DenseNet169
             training. We froze all layers except the last five and
             finalized a starting learning rate of 1e-2, although we
             experimented with 1e-3, 1e-4, and 1e-5. We also used a
             Reduce on Plateau scheduler with a factor of 0.1 and
             patience of six epochs. Again, we used scikit-learn's
             compute_class_weights to adjust for the imbalanced dataset
             with a Cross Entropy Loss function. We utilized torchcam's
             GradCAM on the last feature layer in order to get a better
             understanding of the model's behaviour. Early stopping was
             triggered after 25 epochs due to stagnating loss and F1-Score
             on the validation set.""")

    st.image('images/densenet_training_history.png')

    st.image('images/densenet_grad_cam.png')

    st.write('<h2>ResNet152</h2>', unsafe_allow_html = True)

    st.write("""ResNet152 was trained in the same way as DenseNet152, except
             we used a batch size of 256 due to the larger model size and its'
             impact on GPU limitations. We also started at a learning rate
             of 1e-4. Early stopping was triggeered after 14 epochs due to
             a gradually increasing validation loss.""")

    st.image('images/resnet_training_history.png')

    st.image('images/resnet_grad_cam.png')

    st.write('<h3>Discussion</h3>', unsafe_allow_html = True)

    st.write("""We compared the performance between a DenseNet169, which has
             the advantage of densely connected layers and a lightweight
             model and ResNet152, which is a larger, deeper model, facilitated
             by skip connections. We found that ResNet152 was able to better
             able to classify these images, likely due to the deeper structure,
             which was important for the level of complexity of this image
             classification task. ResNet152 also tends to perform better with
             larger datasets, like the one we have here.""")
             
