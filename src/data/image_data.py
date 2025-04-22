import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample
from sklearn.manifold import LocallyLinearEmbedding
from PIL import Image
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding
from image_functions import plot_components, compute_image_hash, image_processing
from text_functions import safe_loader

def format_data():
    """
    loads in the data, takes a random selection of the data, and
    displays the images and saves them.
    """
    X = pd.read_csv('data/raw/X_train.csv', index_col = 0)
    y = pd.read_csv('data/raw/y_train.csv', index_col = 0)

    X['image'] = X.apply(lambda row: f"image_{row['imageid']}_product_{row['productid']}.jpg", axis = 1)

    data = X[['image']].copy()
    data['target'] = y['prdtypecode']

    random_display_data = pd.concat([
        resample(data[data['target'] == label], replace = False, n_samples = 2, random_state = 42)
        for label in data['target'].unique()])

    random_display_data.sort_values(by = 'target', inplace = True)

    path = 'data/images/image_train'

    fig, ax = plt.subplots(9, 5, figsize = (15, 27), subplot_kw = dict(xticks=[], yticks=[]))
    for i, axis in enumerate(ax.flat):
        with Image.open(f"{path}/{random_display_data['image'].iloc[i]}") as img:
            axis.imshow(img)
        axis.set_ylabel(f"{random_display_data['target'].iloc[i]}", rotation = 0)
        axis.yaxis.label.set_ha('right')

    plt.tight_layout();
    plt.savefig('images/image_samples.png')

    """
    Resamples 30 images of each datapoint for LLE - a dimension reduction technique - that can help
    visualize differences between classes.
    """
    min_count = data['target'].value_counts().min()

    equal_data = pd.concat([
        resample(data[data['target'] == label], replace = False, n_samples = min_count, random_state = 42)
        for label in data['target'].unique()
    ])

    subset_data = pd.concat([
        resample(equal_data[equal_data['target'] == label], replace = False, n_samples = 30,
                 random_state = 42)
        for label in equal_data['target'].unique()
    ])

    """
    Flattens all the images to a 1D array for LLE.
    """
    images = []

    for index, item in enumerate(subset_data['image']):
        with Image.open(f"{path}/{item}") as img:
            img_array = np.array(img)
            images.append(img_array)

    images_flattened = np.array([img.flatten() for img in images])

    lle = LocallyLinearEmbedding(n_neighbors = 50, n_components = 2, method=  'modified',
                                 random_state = 42, n_jobs = -1)

    dataLLE = lle.fit_transform(images_flattened) 

    """
    this takes the flattened dataframe and reshapes it to 500 x 500 x 3.
    """
    image_height, image_width, image_channels = 500, 500, 3
    reshaped_images = images_flattened.reshape((-1, image_height, image_width, image_channels))

    fig, ax = plt.subplots(figsize = (15, 5))
    plot_components(data = dataLLE, model = lle, images = reshaped_images, cmap = subset_data['target'],
                    ax = ax, thumb_frac = 0.05, prefit = True, zoom = 0.08)
    plt.tight_layout();
    plt.savefig('images/LLE.png')

    for index, image in enumerate(data['image']):
        with Image.open(f"{path}/{image}") as img:
            data.loc[index, 'sha256'] = compute_image_hash(img)

    data.to_csv('data/processed/images.csv', index = False)

    print(f"There are {data['sha256'].duplicated().sum()} duplicate images "
          f"- or {round(data['sha256'].duplicated().sum()/len(X), 3)*100}% of the data.")

    duplicated = data[data['sha256'].duplicated()]

    duplicates = duplicated.groupby('sha256').agg({'target' : 'nunique'})

    """
    note that the majority of the duplicates belong have only one target.
    """
    print(duplicates.describe())

    dupes = duplicates[duplicates['target'] > 1].reset_index()
    """
    the majority of the items with more than one target have 2 codes.
    """

    print(dupes.describe())

    X_duplicates = data[data['sha256'].isin(dupes['sha256'].to_list())]
    dupe_images = resample(X_duplicates, replace = False, n_samples = 9)
    dupes2 = X_duplicates.groupby('sha256').agg({'target' : 'unique'}).reset_index()
    dupes3 = dupes2[dupes2['sha256'].isin(dupe_images['sha256'].to_list())]
    dupes3 = dupes3.set_index('sha256').reindex(dupe_images['sha256']).reset_index()

    fig, ax = plt.subplots(3, 3, figsize = (15, 15), subplot_kw = dict(xticks=[], yticks=[]))

    for i, axis in enumerate(ax.flat):

        with Image.open(f"{path}/{dupe_images['image'].iloc[i]}") as img:
            axis.imshow(img)
        axis.set_title(f"{dupes3['target'].iloc[i]}")

    plt.tight_layout();
    plt.savefig('images/duplicates.png')

    duplicate = duplicated.groupby('sha256').agg({'target' : 'unique'})

    items = []

    for row in range(len(duplicate)):

        temp_list = duplicate['target'].iloc[row]

        for item in temp_list:
            items.append(item)

    items_df = pd.DataFrame(items, columns = ['target'])

    fig = plt.figure(figsize = (15, 5))

    sns.countplot(data = items_df, x = 'target')
    plt.title('Number of Unique duplicates per Target')
    plt.xlabel('Target');
    plt.tight_layout()
    plt.savefig('images/distribtion_of_duplicates.png')

    """
    just selecting a small amount of the dataframe for plotting, and demonstration of the function.
    """

    small_data = resample(data, replace = False, n_samples = 9)
    results = small_data.apply(image_processing, axis = 1)
    df = pd.DataFrame(results)
    new_df = pd.DataFrame(df[0].tolist(), columns = ['image_named', 'ratio', 'ratios',
                                                     'coords', 'xycoords', 'img'])
    small_data.reset_index(drop = True, inplace = True)
    new_small_data = pd.concat([small_data, new_df], axis = 1).drop(columns = ['image_named'])

    fig, ax = plt.subplots(6, 2, figsize = (15, 20))

    item = 0
    i = 0

    while item < 6:
        
        image_path = f"{path}/{new_small_data['image'].iloc[item]}"

        with Image.open(image_path) as img:
            ax[i, 0].imshow(img)
        ax[i, 0].plot(new_small_data['xycoords'].iloc[item][0], new_small_data['xycoords'].iloc[item][1],
                      color = 'purple', linestyle = 'dashed', linewidth = 2)
        ax[i, 0].set_ylabel(f"{new_small_data['target'].iloc[item]}", rotation = 0, ha = 'right')
        ax[i, 0].set_xticks([])
        ax[i, 0].set_yticks([])

        if i == 0:
            ax[i, 0].set_title('Original Image')
        item += 1
        i +=1

    item = 0
    i = 0

    while item < 6:
        
        ax[i, 1].imshow(new_small_data['img'].iloc[item])
        ax[i, 1].set_xticks([])
        ax[i, 1].set_yticks([])
        
        if i == 0:
            ax[i, 1].set_title('Resized Image')
        item += 1    
        i +=1

    plt.tight_layout()
    plt.savefig('images/reshaped_examples.png')

    """
    loads in the train, val, test indices for splitting and then drops the irrelevant
    columns from each dataset and saves them for model training.
    """

    train_indices = safe_loader('data/processed/train_indices.pkl')
    val_indices = safe_loader('data/processed/val_indices.pkl')
    test_indices = safe_loader('data/processed/test_indices.pkl')

    train_data = data.iloc[train_indices]
    val_data = data.iloc[val_indices]
    test_data = data.iloc[test_indices]

    split_data = {'train' : train_data,
                  'val' : val_data,
                  'test' : test_data}

    splits = {}

    for split in split_data.keys():
        df = split_data[split]
        df = df[~df['sha256'].isin(dupes['sha256'])]
        df = df.drop_duplicates(subset=['sha256'])
        df = df.drop(columns=['sha256'])
        splits[split] = df

    splits['train'].to_csv('data/processed/train_CNN.csv', index = False)
    splits['val'].to_csv('data/processed/validation_CNN.csv', index = False) 
    splits['test'].to_csv('data/processed/test_CNN.csv', index = False)   

if __name__ == '__main__':
    format_data()
