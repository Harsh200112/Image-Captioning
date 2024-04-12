import os
import re
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

max_caption_length = 28
image_size = 224
batch_size = 32


def lower_case(text):
    l = []
    for i in text.split():
        l.append(i.lower())

    return " ".join(l)


def remove_numbers_punctuation_special(text):
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation and special symbols
    text = re.sub(r'[^\w\s]', '', text)
    return text


class ImagesToCaptionDataset(Dataset):
    def __init__(self, caption_path, images_path, max_caption_length, frequency_threshold, transform=None):
        super().__init__()
        self.frequency_threshold = frequency_threshold
        self.images_path = images_path
        self.data = pd.read_csv(caption_path)
        self.images = self.data['image']
        self.captions = self.data['caption']
        self.captions = self.captions.apply(lower_case)
        self.captions = self.captions.apply(remove_numbers_punctuation_special)
        self.transform = transform
        self.max_length = max_caption_length - 2
        self.vocabulary = self.get_vocabulary()
        self.vocab_size = len(self.vocabulary)

    def get_vocabulary(self):
        vocab = {'<sos>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3}
        frequency = {}
        idx = 4
        for i in self.captions:
            for j in i.split():
                if j not in frequency:
                    frequency[j] = 1
                else:
                    frequency[j] += 1

                if frequency[j] == self.frequency_threshold:
                    vocab[j] = idx
                    idx += 1
        return vocab

    def prepare_sequence(self, seq):
        idxs = [0]
        idxs += [self.vocabulary[w] if w in self.vocabulary else self.vocabulary['<unk>'] for w in seq]
        idxs += [2]

        if len(idxs) < self.max_length + 2:
            idxs += [1] * (self.max_length + 2 - len(idxs))

        else:
            idxs = idxs[:self.max_length + 2]

        return idxs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_name = self.images[index]
        image = Image.open(os.path.join(self.images_path, image_name)).convert('RGB')
        caption = self.captions[index]
        caption = self.prepare_sequence(caption.split())

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(caption, dtype=torch.long)


transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
])

dataset = ImagesToCaptionDataset('C:/Users/Harsh Soni/Downloads/DL Project Captioning/flickr8k/captions.txt', 'C:/Users/Harsh Soni/Downloads/DL Project Captioning/flickr8k/images', max_caption_length,
                                 5,
                                 transform)
vocabulary_size = dataset.vocab_size
caption_vocab = dataset.vocabulary

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.7, 0.2, 0.1])

train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size, shuffle=True)
