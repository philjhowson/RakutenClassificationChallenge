import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import text_functions as textfunc

def format_data():

    df_X = pd.read_csv('data/raw/X_train.csv', index_col = 0, encoding = 'utf8')
    df_Y = pd.read_csv('data/raw/y_train.csv', index_col = 0, encoding = 'utf8')
    df = pd.concat([df_X, df_Y[['prdtypecode']] ], axis=1)
    df['initial_index'] = df.index

    df['description'] = df['description'].fillna(str())
    df['description'] = df['description'].astype('str')

    df['image_name'] = 'image_' + df['imageid'].astype(str) + '_product_' + df['productid'].astype(str)

    filter_func_list = [textfunc.filter_nofilter, textfunc.filter_ftfy,
                        textfunc.filter_unicodedata, textfunc.filter_bs4,
                        textfunc.filter_norm, textfunc.filter_username,
                        textfunc.filter_whitespace]

    filtered_column_list = [textfunc.base_name_func(filter_func).split( '_')[1] for filter_func in filter_func_list]

    column_func_list = [textfunc.filter_designation, textfunc.filter_description]
    column_list = [textfunc.base_name_func(column_func).split( '_')[1] for column_func in column_func_list]
    metric_func_list = [textfunc.str_length, textfunc.str_wordcount]
    metric_list = [textfunc.base_name_func(metric_func).split( '_')[1] for metric_func in metric_func_list]

    print( 'start with filters into the columns' )
    for i, column in enumerate(column_list):
        for j, (filtered_column, filter_func) in enumerate(zip(filtered_column_list, filter_func_list) ):
            new_column = str()
            old_column = str()

            if filtered_column == filtered_column_list[0]:
                old_column = column
                new_column = column + '_' + filtered_column

            else:
                old_column = column + '_' + filtered_column_list[ j-1 ]
                new_column = column + '_' + filtered_column   

            df.loc[:,new_column] = df.loc[:,old_column].apply( filter_func )

            new_column_altered = new_column + '_altered'
            df.loc[:,new_column_altered] = (df[new_column] != df[old_column])

            for k, (metric, metric_func) in enumerate(zip( metric_list, metric_func_list ) ):
                new_column_metric = new_column + '_' + metric
                df.loc[:,new_column_metric] = df.loc[:,new_column].apply( metric_func )


            new_column_metric = new_column + '_' + 'duplicates'
            old_column_metric = old_column + '_' + 'duplicates'
            
            if filtered_column == 'nofilter':
                df.loc[:,new_column_metric] = df[column][ df[column] != '' ][df[column][ df[column] != '' ].duplicated(keep=False)].shape[0]
            else:
                df.loc[:,new_column_metric] = df[new_column][ df[new_column] != '' ][df[new_column][ df[new_column] != '' ].duplicated(keep=False)].shape[0]

    print('Finished with filters into the columns')

    print('Start applying stacked filters')

    df.loc[:, 'designation_filtered'] = df.loc[:, 'designation'].apply(
        lambda text: textfunc.filter_stacked(text, filter_func_list))
    df.loc[:,'description_filtered'] = df.loc[:,'description'].apply(
        lambda text: textfunc.filter_stacked(text, filter_func_list))

    print('Done filtering')

    metric_list_all = metric_list + ['altered', 'duplicates']
    row = len(column_list)
    col = len(metric_list_all)

    fig, axs = plt.subplots(row, col, constrained_layout = True, figsize = (10,8))

    for i, column in enumerate( column_list ):

        filter_func_list_column = filter_func_list

        filtered_column_list_column = filtered_column_list
        

        for j, metric in enumerate(metric_list_all):

            plot_column = []

            for k, filtered_column in enumerate( filtered_column_list ):
                plot_column.append( column + '_' + filtered_column + '_' + metric )
                
            if j<2:
                axs[i,j].plot( filtered_column_list_column,df[plot_column].mean().values, label = 'mean')
                axs[i,j].plot( filtered_column_list_column,df[plot_column].min().values, label = 'min')
                axs[i,j].plot( filtered_column_list_column,df[plot_column].max().values, label = 'max')
            elif j <3:
                axs[i,j].plot( filtered_column_list_column,df[plot_column].sum().values, label = 'count')
            else:
                axs[i,j].plot( filtered_column_list_column,df[plot_column].max().values, label = 'count')

            for tick in axs[i,j].get_xticklabels():
                tick.set_rotation(75)
            axs[i,j].set_xlabel( 'applied procedure')
            axs[i,j].set_ylabel( metric )
            axs[i,j].set_title( column )
            axs[i,j].legend()

    plt.savefig('images/filtering_results.png')

    duplicates = df[['designation','prdtypecode']][ df['designation'] != '' ][df['designation'][ df['designation'] != ''].duplicated(keep=False)]['prdtypecode'].value_counts()
    plt.xticks(rotation = 60)
    plt.title('duplicates: designation before filtering')
    sns.barplot(duplicates, color = 'blue')
    plt.tight_layout()
    plt.savefig('images/designation_duplicates_before_filter.png')

    duplicates = df[['designation_filtered','prdtypecode']][ df['designation_filtered'] != '' ][df['designation_filtered'][ df['designation_filtered'] != ''].duplicated(keep=False)]['prdtypecode'].value_counts()
    plt.xticks(rotation = 60)
    plt.title('duplicates: designation after filtering')
    sns.barplot(duplicates, color = 'purple')
    plt.tight_layout()
    plt.savefig('images/designation_duplicates_after_filter.png')

    duplicates = df[['description','prdtypecode']][ df['description'] != '' ][df['description'][ df['description'] != '' ].duplicated(keep=False)]['prdtypecode'].value_counts()
    plt.xticks(rotation = 60)
    plt.title('duplicates: description before filtering')
    sns.barplot(duplicates, color = 'blue')
    plt.tight_layout()
    plt.savefig('images/description_duplicates_before_filter.png')

    duplicates = df[['description_filtered','prdtypecode']][ df['description_filtered'] != '' ][df['description_filtered'][ df['description_filtered'] != ''].duplicated(keep=False)]['prdtypecode'].value_counts()
    plt.xticks(rotation = 60)
    plt.title('duplicates: description after filtering')
    sns.barplot(duplicates, color = 'purple')
    plt.tight_layout()
    plt.savefig('images/description_duplicates_after_filter.png')

    df_filtered = df_filtered[['designation_filtered', 'description_filtered', 'image_name', 'initial_index', 'prdtypecode']].copy()
    df_translated = textfunc.detect_and_translate_offline(df_filtered)
    df_translated.to_parquet('data/processed/translated_text.parquet')

    fig, ax = plt.subplots(1, 2, figsize = (20, 10))

    sns.countplot(data = df_translated, x = 'designation_lang', ax = ax[0])
    sns.countplot(data = df_translated, x = 'description_lang', ax = ax[1])
    ax[0].set_xlabel('Language')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Designation')
    ax[1].set_xlabel('Language')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Description')
    fig.suptitle('Language Counts prior to Translation')
    plt.tight_layout()
    plt.savefig('images/pretranslation_languages.png')

    fig, ax = plt.subplots(1, 2, figsize = (20, 10))

    sns.countplot(data = df_translated, x = 'designation_lang_after', ax = ax[0])
    sns.countplot(data = df_translated, x = 'description_lang_after', ax = ax[1])
    ax[0].set_xlabel('Language')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Designation')
    ax[1].set_xlabel('Language')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Description')
    fig.suptitle('Language Counts after to Translation')
    plt.tight_layout()
    plt.savefig('images/posttranslation_languages.png')

if __name__ == '__main__':
    format_data()
