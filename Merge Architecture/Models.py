import torch
import torch.nn as nn
from torchvision import models


class ImageEmbedding(nn.Module):
    def __init__(self, hidden_size, embedding_size):
        super().__init__()
        model = models.resnet152()
        model.load_state_dict(torch.load('resnet152-b121ed2d.pth'))
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, hidden_size),
            nn.Linear(hidden_size, embedding_size)
        )
        self.model = model

    def forward(self, x):
        return self.model(x)


# x1 = torch.rand(32, 3, 224, 224)
# model = ImageEmbedding(1024, 512)
# print(model(x1).shape)


class TextEmbedding(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=2*hidden_size)

    def forward(self, x):
        embeddings = self.embeddings(x)
        embeddings = self.dropout(embeddings)
        outs1, _ = self.lstm1(embeddings)
        outs1 = self.dropout(outs1)
        outs2, _ = self.lstm2(outs1)
        return outs2


# x2 = torch.randint(0, 1000, (32, 25))
# model = TextEmbedding(50, 1000, 128)
# print(model(x2).shape)


class ImageCaptioner(nn.Module):
    def __init__(self, image_embed_size, text_embed_size, vocab_size, hidden_size, image_hid_size):
        super().__init__()
        self.image_embeddings = ImageEmbedding(image_hid_size, image_embed_size)
        self.text_embeddings = TextEmbedding(text_embed_size, vocab_size, hidden_size)
        self.words = nn.Sequential(
            nn.Linear(2*hidden_size, 16*hidden_size),
            nn.Linear(16*hidden_size, vocab_size)
        )

    def forward(self, image, captions):
        image_embeddings = self.image_embeddings(image)
        text_embeddings = self.text_embeddings(captions)
        # print(image_embeddings.shape, text_embeddings.shape)
        image_embeddings = image_embeddings.unsqueeze(1).expand_as(text_embeddings)
        total_embeddings = torch.add(image_embeddings, text_embeddings)
        return self.words(total_embeddings)


# model = ImageCaptioner(512, 256, 1000, 256, 1024)
# print(model(x1, x2).shape)

# x1 = torch.rand(32, 25, 256)
# x2 = torch.rand(32, 25, 256)
#
# x1_flat = x1.view(x1.size(0), -1)  # reshape to (32, 25*256)
# x2_flat = x2.view(x2.size(0), -1)  # reshape to (32, 25*256)
#
# # Normalize the embeddings
# x1_norm = torch.nn.functional.normalize(x1_flat, p=2, dim=1)
# x2_norm = torch.nn.functional.normalize(x2_flat, p=2, dim=1)
#
# # Calculate cosine similarity
# similarity_matrix = torch.matmul(x1_norm, x2_norm.T)
#
# print(similarity_matrix.shape)
