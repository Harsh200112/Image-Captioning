import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


class ImageEmbedding(nn.Module):
    def __init__(self, embedding_size, caption_length):
        super().__init__()
        self.caption_length = caption_length
        self.conv = nn.Sequential(
            ConvBlock(3, 32, 3, 1),
            ConvBlock(32, 64, 3, 1),
            ConvBlock(64, 128, 3, 1)
        )
        self.embedding = nn.Linear(3584, embedding_size//2)
        self.lstm = nn.LSTM(embedding_size//2, embedding_size)

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape(x.shape[0], self.caption_length, -1)
        x = self.embedding(x)
        outs, _ = self.lstm(x)
        return outs


# x1 = torch.rand(32, 3, 224, 224)
# model = ImageEmbedding(256, 28)
# print(model(x1).shape)


class TextEmbedding(nn.Module):
    def __init__(self, embedding_size, vocab_size, hidden_size, max_caption_length):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(0.2)
        self.lstm1 = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=2*hidden_size)
        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size * max_caption_length, 100),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        embeddings = self.embeddings(x)
        embeddings = self.dropout(embeddings)
        outs1, _ = self.lstm1(embeddings)
        outs1 = self.dropout(outs1)
        outs2, _ = self.lstm2(outs1)
        outs2 = outs2.reshape(outs2.shape[0], -1)
        return self.linear(outs2)


# x2 = torch.randint(0, 1000, (32, 28))
# model = TextEmbedding(50, 1000, 128, 28)
# print(model(x2).shape)
