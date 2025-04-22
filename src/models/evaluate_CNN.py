import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchcam.methods import GradCAM
from torchvision import transforms, models
from sklearn.metrics import f1_score, classification_report
from CNN_custom_functions import ImageDataset, resizing
from text_custom_functions import safe_loader, safe_saver
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import argparse


def resize_image(img):
    return resizing(img, 224)

def evaluate_model(cnn = None):
    """
    this function imports a trained model and then evaluates it using
    f1 metrics, creating an image of training history, and also creates
    a gradcam image.
    """

    """
    loads in the previously created test split, and then converts all the
    labels to numbers 0-26, so that the labels can be correctly interpreted
    by the CNN. Saves the label pairs as a dictionary.
    """
    test = pd.read_csv('data/processed/test_CNN.csv')
    image_labels = test['target'].copy()
    unique_labels = sorted(set(test['target']))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    test['target'] = test['target'].map(label_to_index)

    with open('data/processed/test_label_dictionary.json', 'w') as f:
        json.dump(label_to_index, f)

    """
    sets up the transforms, in the test case, this is just normalizations,
    and the custom resizing. Then creates the data loaders.
    """

    img_dir = 'data/images/image_train/'

    test_transform = transforms.Compose([
        transforms.Lambda(resize_image),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])

    test_data = ImageDataset(categories = test, img_dir = img_dir,
                             transform = test_transform)

    test_loader = DataLoader(test_data, batch_size = 256, shuffle = False,
                             num_workers = 4)

    """
    loads the selected model and model weights, based on the input argument.
    Freezes all layers except the last because gradient calculations are
    needed for gradcam. Moves the device to the GPU if a GPU exists.
    """

    if cnn == 'resnet':

        model = models.resnet152(weights = None)
        model.fc = nn.Linear(model.fc.in_features, 27)
        model.load_state_dict(torch.load('models/resnet_model_weights.pth',
                                         weights_only = True))
        
        for param in model.parameters():
            param.requires_grad = False

        for param in model.layer4.parameters():
            param.requires_grad = True

    elif cnn == 'densenet':

        model = models.densenet169(weights = None)
        model.classifier = nn.Linear(model.classifier.in_features, 27)
        model.load_state_dict(torch.load('models/densenet_model_weights.pth',
                                         weights_only = True))

        for param in model.parameters():
            param.requires_grad = False

        layer = model.features.denseblock4.denselayer32

        for param in layer.parameters():
            param.requires_grad = True

    else:
        
        print("Incompatible -cnn argument. Please choose 'densenet' or 'resnet'")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    """
    sets model to eval and then loads images and labels. Records each
    prediction and each true label for each batch. Saves the model metrics.
    """

    model.eval()

    test_preds = []
    test_labs = []

    for test_images, test_labels in test_loader:

        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test_outputs = model(test_images)

        predicted_classes = torch.argmax(test_outputs, dim = 1)
        test_preds.extend(predicted_classes.cpu().numpy())
        test_labs.extend(test_labels.cpu().numpy())

    test_f1 = f1_score(test_labs, test_preds, average = 'weighted')
    test_report = classification_report(test_labs, test_preds, output_dict = True)

    safe_saver(test_report, f'metrics/{cnn}_classification_report.pkl')

    history = safe_loader(f'metrics/{cnn}_performance.pkl')

    values = [max(history['f1']), max(history['val_f1']), test_f1]
    labels = ['Training F1', 'Validation F1', 'Test F1']

    """
    generates a bar plot for each of the training, validation, and test
    f1-scores. Saves the plots.
    """

    plt.figure(figsize = (10, 5))

    bars = plt.bar(labels, values, color = ['blue', 'purple', 'green'])

    for bar in bars:
        yval = round(bar.get_height(), 3)
        plt.text(bar.get_x() + bar.get_width() / 2, yval - 0.05, str(yval),
                 ha = 'center', va = 'bottom', color = 'white', fontweight = 'bold')

    plt.ylim(0, 1)
    plt.ylabel('F1-Score')
    plt.title('Training, Validation, and Test F1-Scores')
    plt.tight_layout()

    plt.savefig(f'images/{cnn}_f1_scores.png')

    training_items = ['loss', 'f1', 'gradient']
    validation_items = ['val_loss', 'val_f1']
    titles = ['Loss', 'F1-Score', 'Gradient']

    ticks = [1]
    for tick in list(range(1, len(history['loss']) + 1)):
        if tick % 5 == 0:
            ticks.append(tick)

    fig, ax = plt.subplots(1, 3, figsize = (20, 5))

    for index, axes in enumerate(ax.flat):
        
        if index == 2:
            axes.plot(range(1, len(history[training_items[index]]) + 1),
                      history[training_items[index]], label = 'Training')
            axes.set_xticks(ticks)
            axes.set_title(f"{titles[index]} by Epoch")
            axes.set_xlabel('Epoch')
            axes.set_ylabel(f"{titles[index]}")
            break

        axes.plot(range(1, len(history[training_items[index]]) + 1),
                  history[training_items[index]], label = 'Training')
        axes.plot(range(1, len(history[validation_items[index]]) + 1),
                  history[validation_items[index]], label = 'Validation')
        axes.set_xticks(ticks)
        axes.set_title(f"{titles[index]} by Epoch")
        axes.set_xlabel('Epoch')
        axes.set_ylabel(f"{titles[index]}")
        axes.legend()

    plt.tight_layout()
    plt.savefig(f'images/{cnn}_training_history.png')

    """
    creates a gradcam object and extracts the feature maps for a set of
    images, assigns the model predicted label, the model certainty
    (probability) and the actual label to a list for plotting.
    """

    if cnn == 'resnet':

        grad_cam = GradCAM(model, model.layer4[-1].conv3)

    if cnn == 'densenet':

        grad_cam = GradCAM(model, model.features.denseblock4.denselayer32.conv2)

    #sample_index = [7150, 6752, 2916, 1410, 7820, 4598, 4928, 1084, 4725, 3365]
    sample_index = random_indices = np.random.choice(len(test), size = 10, replace = False)
    print(sample_index)
    sample_images = test['image'].iloc[sample_index].values
    labels = image_labels.iloc[sample_index].values

    activations = []
    predicted_labels = []
    probability = []

    for index, img in enumerate(sample_images):

        img = Image.open(f"{img_dir}{img}")

        input_tensor = test_transform(img).unsqueeze(0).to(device)

        with torch.enable_grad():
            out = model(input_tensor)

        probabilities = nn.functional.softmax(out, dim = 1)
        predicted_class_idx = torch.argmax(probabilities, dim = 1).item()
        predicted_class_prob = probabilities[0, predicted_class_idx].item()
        probability.append(predicted_class_prob)

        predicted_label = next((key for key, value in label_to_index.items() if value == predicted_class_idx), None)
        predicted_labels.append(predicted_label)
        
        activation_map = grad_cam(0, out)
        activations.append(activation_map)

    grad_cam.remove_hooks()

    transform = transforms.Compose([transforms.Lambda(resize_image),
                                    transforms.ToTensor()])

    fig, ax = plt.subplots(2, 5, figsize = (20, 10))

    for index, axes in enumerate(ax.flat):

        with Image.open(f"{img_dir}{sample_images[index]}") as img:
            img = transform(img)

        """
        normalizes the feature maps and resizes them to the same size
        as the input image. Then the feature map and the original image
        are combined.
        """

        activation_map = activations[index]
        activation_map = activation_map[0].cpu().squeeze()

        activation_map = activation_map - activation_map.min()
        activation_map = activation_map / activation_map.max()
        activation_map = activation_map.unsqueeze(0).unsqueeze(0)

        activation_map_resized = F.interpolate(activation_map, size = (224, 224),
                                               mode = 'bilinear', align_corners = False)

        heatmap = plt.cm.jet(activation_map_resized.squeeze().cpu().numpy())
        heatmap = np.delete(heatmap, 3, axis = -1)

        heatmap /= (heatmap.max() + 1e-8)
        image = img.permute(1, 2, 0).numpy()

        overlayed_image = (image * (1 - 0.5) * 255 + heatmap * 0.5 * 255).astype(np.uint8)

        axes.imshow(overlayed_image)
        axes.set_title(f"Predicted: {predicted_labels[index]} ({round(probability[index] * 100, 2)}%),\n Actual: {labels[index]}")

    plt.tight_layout()

    plt.savefig(f'images/{cnn}_grad_cam.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Evaluate a trained model.')
    parser.add_argument('--cnn', type = str, required = True, help = 'either resnet or densenet')

    args = parser.parse_args()
    evaluate_model(args.cnn)
    

        

    
