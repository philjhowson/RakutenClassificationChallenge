# Rakuten Classification Challenge
***

This project uses a multi-modal approach to the classification problem posed by Rakuten:

"The goal of this data challenge is large-scale multimodal (text and image) product data
classification into product type codes.

For example, in Rakuten France catalog, a product with a French designation or title
Klarstein Présentoir 2 Montres Optique Fibre associated with an image and sometimes with
an additional description. This product is categorized under the 1500 product type code.
There are other products with different titles, images and with possible descriptions,
which are under the same product type code. Given these information on the products, like
the example above, this challenge proposes to model a classifier to classify the products
into its corresponding product type code."

The challenge data is not hosted on this repository. For access to challenge data, please
visit the challenge site below, register, and then download the data.

[Rakuten Multimodal Product Data Classification](https://challengedata.ens.fr/challenges/35)

## Project Organization
------------------------------------------------------------------------
    root
    ├── data # contains image and text data, not stored on GitHub due to size limitations
    │   ├── images # contains images
    │   │    ├── image_test # test data for submissions
    │   │    └── image_train # the data used for model training and validation
    │   ├── processed # processed data files
    │   └── raw # the raw text file from the rakuten challenge
    ├── images # contains output images from data exploration and model evaluation
    ├── metrics # output metrics for model training and evaluation
    ├── models # output folder for model saves, this is not on github due to model sizes
    ├── src # contains source code for exploration, not stored on GitHub due to size limitations
    │   ├── data # code for data formatting
    │   │    ├── create_train_test_split.py # creates the training, test, and validation indicies
    │   │    ├── image_data.py # describes and formats images
    │   │    ├── image_function.py # custom functions for use in image_data.py
    │   │    ├── text_data.py # describes, formats, and translates text
    │   │    ├── text_formatting.py # some final formatting, stop word removal etc. for text
    │   │    └── text_functions.py # custom functions used during formatting
    │   └── models # code for training models
    │        ├── CNN_custom_functions.py # custom functions used for training CNNs
    │        ├── densenet.py # training DenseNet169
    │        ├── evaluate_CNN.py # evaluates the CNN models
    │        ├── evaluated_text.py # evaluates the text models
    │        ├── model_comparison.py # compares F1-Scores for each model
    │        ├── multimodal.py # trains text/image model
    │        ├── resnet.py # trains ResNet152
    │        ├── roBERTa.py # trains roBERTa-base on English data
    │        ├── roBERTa_multilang.py # trains roBERTa-base on multilingual data
    │        └── text_custom_functions.py # custom functions used for text processing
    ├── streamlit # contains the files for the streamlit app
    │   ├── images # contains images used in the streamlit app
    │   └── items # contains src files for the streamlit app 
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt

## Project Introduction

