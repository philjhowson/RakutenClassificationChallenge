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

The Streamlit app can be found here:

[Rakuten Classification Challenge Streamlit App](https://rakutenclassificationchallenge.streamlit.app/)

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
The goal of this project is to classify products using text and image data. Like most real
datasets it requires careful filtering of the data to remove all the noise and properly 
categorize the products.

We began by filtering the text data to remove noisy components HTML tags, trailing and 
leading white spaces, non-word characters, usernames, duplicate strings, punctuation,
and by converting unicode and hexcode to the appropriate characters. We further translated
text into English so we could use roBERTa-base trained on English day to compare the performance
with roBERTa-base multilingual.

Images were also analyzed to determine the distribution in images across categories,
remove duplicates, and remove whitespace in the images and maximize the image size.
We also resized them to an appropriate size for DenseNet169 and ResNet152.

We trained each model individually on the corresponding text and images to ascertain
their performance before combining text and image models into a multimodal classifier
through feature fusion.

We achieve a best test score of 0.883 with a feature fusion approach using DenseNet169
for images and roBERTa-base multilingual for text.

## Getting Started
Clone the repository:

```sh
git clone https://github.com/philjhowson/RakutenClassificationChallenge
```

Download the data from https://challengedata.ens.fr/challenges/35 and put the raw
data in a folder /data/raw.

It is ideal to run this in a virtual environment and install the relevant dependencies.

```sh
python -m venv my_env
my_env/Scripts/activate
pip install -r requirements.txt
```

If you run the scripts as __main__ as intended, they all assume you are in the
root directory. The scripts in running order are:

1. create_train_test_split.py # creates the indices for training, validation, and test
2. text_data.py # this script translates and filters text, as such it takes a long time to run
3. text_formatting.py # fixes a few unicode/hexcode items and removes stop words and lemmatizes words
4. image_data.py # performs the exploration and formatting of images

After those scripts, models can generally be trained in any order, but the single models need
to be trained before multimodal models. The single model scripts are:

1. densenet.py
2. resnet.py
3. roBERTa.py
4. roBERTa_multilang.py

To evaluate the CNN models, there is an expected argument --image. This takes the model you want
to evaluate, either densenet or resnet.

```sh
python src/models/evaluate_CNN.py --image densenet
```

The same is true for the transformer models, except there is an expected argument --text. This
takes the model you want to evaluate, either english or multi.

```sh
python src/models/evaluate_text.py --text english
```

When training the multimodal modal, both the text and image models need to be specified. This is
done with the --text and --image arguments that take the same values as above.

```sh
python src/models/multimodal.py --text multi --image densenet
```

The evaluation of the multimodal models is performed the same way, with both models needing to
be specified with the --text and --image arguments.

The final script model_comparison.py compares all the models and assumes you have trained all
the models as it imports the classification reports for all the models to make the bar plot.

## Future Directions

Our results could likely be improved upon with more careful filtering and randomized
text transformations similar to the transforms.Compose() in order to reduce overfitting
and enhance generalizations.