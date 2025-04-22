from text_custom_functions import safe_loader
import matplotlib.pyplot as plt
import seaborn as sns

def model_comparison():
    """
    this function just loads in all the saved classification reports
    and pulls out the test f1 score for each and plots them in a barplot.
    """
    densenet_history = safe_loader('metrics/densenet_classification_report.pkl')
    resnet_history = safe_loader('metrics/resnet_classification_report.pkl')
    roBERTa_history = safe_loader('metrics/roBERTa_classification_report.pkl')
    roBERTa_multi_history = safe_loader('metrics/roBERTa_multi_classification_report.pkl')
    roBERTa_densenet_history = safe_loader('metrics/roBERTa+densenet_classification_report.pkl')
    roBERTa_resnet_history = safe_loader('metrics/roBERTa+resnet_classification_report.pkl')
    roBERTa_multi_densenet_history = safe_loader('metrics/roBERTa_multi+densenet_classification_report.pkl')
    roBERTa_multi_resnet_history = safe_loader('metrics/roBERTa+resnet_classification_report.pkl')    

    densenet_f1 = densenet_history['weighted avg']['f1-score']
    resnet_f1 = resnet_history['weighted avg']['f1-score']
    roBERTa_f1 = roBERTa_history['weighted avg']['f1-score']
    roBERTa_multi_f1 = roBERTa_multi_history['weighted avg']['f1-score']
    roBERTa_densenet_f1 = roBERTa_densenet_history['weighted avg']['f1-score']
    roBERTa_resnet_f1 = roBERTa_resnet_history['weighted avg']['f1-score']
    roBERTa_multi_densenet_f1 = roBERTa_multi_densenet_history['weighted avg']['f1-score']
    roBERTa_multi_resnet_f1 = roBERTa_multi_resnet_history['weighted avg']['f1-score']
    
    f1_values = [densenet_f1, resnet_f1, roBERTa_f1, roBERTa_multi_f1,
                 roBERTa_densenet_f1, roBERTa_resnet_f1, roBERTa_multi_densenet_f1,
                 roBERTa_multi_resnet_f1]
    f1_labels = ['DenseNet F1-Score', 'ResNet F1-Score', 'roBERTa F1-Score',
                 'roBERTa Multilingual F1-Score', 'roBERTa + DenseNet F1-Score',
                 'roBERTa + ResNet F1-Score', 'roBERTa Multilingual + DenseNet F1-Score',
                 'roBERTa Multilingual + ResNet F1-Score']

    colors = sns.color_palette('deep', len(f1_labels))

    plt.figure(figsize = (10, 5))

    bars = plt.bar(f1_labels, f1_values, color = colors)

    for bar in bars:
        yval = round(bar.get_height(), 3)
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.1, str(yval),
                 ha = 'center', va = 'bottom', color = 'white',
                 fontweight = 'bold')

    plt.xticks(rotation = 45, ha = 'right', va = 'top')
    plt.ylim(0, 1)
    plt.ylabel('F1-Score')
    plt.title('Test Score Comparison for each Model')
    plt.tight_layout()

    plt.savefig('images/model_f1_comparisons.png')

if __name__ == '__main__':
    model_comparison()
