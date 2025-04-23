import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def data_description():

    st.write("<h1>Data Description</h1>", unsafe_allow_html = True)

    st.write("<h2>Text Description</h2>", unsafe_allow_html = True)

    st.write("""We found that the Rakuten dataset contained a large
             number of elements that required filtering and adjusting.
             We used ftfy to fix encoding issue, unicodedata to apply
             unicode normalization and remove diacritics, and bs4 to strip
             HTML tags. We additionally remove training and leading
             white space and converted multiple spaces into a single space.
             Finally, we removed usernames from the data as well. Below
             is an image of the character length, word count, and number
             of duplicates before and after each processed.""")

    st.image('images/filtering_results.png')

    st.write("""Post-filtering, we searched for duplicate strings. The
             image below reveals the number of pre- and post-filtering
             duplicate images for each of the text columns.""")

    with open('images/designation_duplicates_before_filter.png', 'rb') as f:
        designation_before = Image.open(f)
        designation_before = np.array(designation_before)

    with open('images/designation_duplicates_after_filter.png', 'rb') as f:
        designation_after = Image.open(f)
        designation_after = np.array(designation_after)

    with open('images/description_duplicates_before_filter.png', 'rb') as f:
        description_before = Image.open(f)
        description_before = np.array(description_before)

    with open('images/description_duplicates_after_filter.png', 'rb') as f:
        description_after = Image.open(f)
        description_after = np.array(description_after)

    fig, ax = plt.subplots(2, 2)

    ax[0, 0].imshow(designation_before)
    ax[0, 0].axis('off')
    ax[0, 1].imshow(designation_after)
    ax[0, 1].axis('off')
    ax[1, 0].imshow(description_before)
    ax[1, 0].axis('off')
    ax[1, 1].imshow(description_after)
    ax[1, 1].axis('off')
    plt.tight_layout()

    st.pyplot(fig)

    st.write("""We also translated the text into English so that we could
             use the roBERTa model that was specially trained on English.
             Below, we present a plot of the distribution of languages
             before and after translation. We used langdetect to identify
             the language in each string.""")

    with open('images/pretranslation_languages.png', 'rb') as f:
        pretranslation = Image.open(f)
        pretranslation = np.array(pretranslation)

    with open('images/posttranslation_languages.png', 'rb') as f:
        posttranslation = Image.open(f)
        posttranslation = np.array(posttranslation)

    fig, ax = plt.subplots(2, 1)

    ax[0].imshow(pretranslation)
    ax[0].axis('off')
    ax[1].imshow(posttranslation)
    ax[1].axis('off')
    plt.tight_layout()

    st.pyplot(fig)

    st.write("<h2>Image Description</h2>", unsafe_allow_html = True)

    st.write("""We began image processing by sampling the images and
             performing an LLE to get an idea of class distribution in
             the feature space. Below is an sample image for each class
             and an LLE to visualize the feature space. The prelimary
             look at the data shows a great deal of variation in size,
             type, and content, as well as a great deal of overlap""")

    st.image('images/image_samples.png')

    st.image('images/LLE.png')

    st.write("""Our first objective was to create a custom resizing function
             to detect whitespace surrounding the image, remove the white space
             and resize the image. We chose 224 x 224, because we planned to
             use DenseNet169, which was trained on the ImageNet dataset. We used
             a conjunction of numpy and cv2 to filter out whitespace and resize
             the images. We took the longest side (either the height or width) and
             resized it to 224, and scaled the shorter side to the correct size to
             avoid stretching and skewing. Then we added whitespace on the shorter
             side as appropraite to create a 224 x 224 image. Below we present
             an image of before and after resizing of images. This function was
             used in the transforms.Compose() portion of model training;
             therefore, images were not saved prior as their resized shapes.""")

    st.image('images/reshaped_examples.png')

    st.write("""We also used sha256 to encode the images for each comparison of
             duplicate images. We found a significant number of duplicate images.
             While most duplicates had only a single pair, we did find that
             there were some images that had multiple duplicates across more
             than one category. Image duplicates that were across categroies were
             marked for removal because the likelihood that it would impact
             model performance. Below we present a plot of the duplicate distribution
             across each of the classes""")

    st.image('images/distribtion_of_duplicates.png')
