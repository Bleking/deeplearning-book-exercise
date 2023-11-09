import torch

from mnist_classifier.models.fc import ImageClassifier
from mnist_classifier.models.cnn import ConvolutionalClassifier


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST('../data', train=is_train, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    x = dataset.data.float() / 255.0
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)
    
    return x, y

def split_data(x, y, train_ratio=0.8):
    train_cnt = int(x.size(0) * train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # shuffle dataset to split into train/valid set
    indices = torch.randperm(x.size(0))
    x = torch.index_select(x, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(y, dim=0, index=indices).split([train_cnt, valid_cnt], dim=0)
    
    return x, y

def get_hidden_sizes(input_size, output_size, n_layers):
    step_size = int((input_size - output_size) / n_layers)

    hidden_sizes = []
    current_size = input_size

    for i in range(n_layers - 1):
        hidden_sizes += [current_size - step_size]
        current_size = hidden_sizes[-1]
    
    return hidden_sizes

def get_model(input_size, output_size, config, device):
    if config.model == 'fc':
        model = ImageClassifier(
            input_size = input_size, 
            output_size = output_size, 
            hidden_sizes = get_hidden_sizes(input_size, output_size, config.n_layers), 
            use_batch_norm = not config.use_dropout, 
            dropout_p = config.dropout_p
        ).to(device)
    elif config.model == 'cnn':
        model = ConvolutionalClassifier(output_size)
    else:
        raise NotImplementedError
    
    return model