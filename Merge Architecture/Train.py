import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Models import ImageCaptioner
from DataLoaders import train_loader, test_loader, val_loader, vocabulary_size, caption_vocab

# Hyperparameters
lr = 3e-4
image_size = 224
batch_size = 64
epochs = 50
embed_size = 512
text_embed_size = 256
hidden_size = 256
image_hid_size = 1024
max_caption_length = 28

reversed_vocab = {value: key for key, value in caption_vocab.items()}


def get_real_captions(reversed_vocab, batch_tensors):
    sentences = []
    for tensor in batch_tensors:
        # Map each number in the tensor to its corresponding word using the reversed dictionary
        words = [reversed_vocab[int(number)] for number in tensor]
        # Join the words to form sentences
        sentence = ' '.join(filter(None, words))  # filter(None, ...) removes None values from the list
        sentences.append(sentence)

    return sentences


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current Device:-", device)

# Directory to save checkpoints
checkpoint_dir = 'merg_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Models
model = ImageCaptioner(embed_size, text_embed_size, vocabulary_size, hidden_size, image_hid_size).to(device)

# Optimizers
optimizer = optim.Adam(model.parameters(), lr)

# Loss Functions
criterion = nn.CrossEntropyLoss(ignore_index=caption_vocab['<pad>'])


# Function to save checkpoint
def save_checkpoint(epoch, model, optimizer, losses, val_losses):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch.pt')
    torch.save({
        'epoch': epoch,
        'Loss': losses,
        'val_loss': val_losses,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')


# Function to load checkpoint
def load_checkpoint(model, optimizer):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        losses = checkpoint['Loss']
        val_losses = checkpoint['val_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Checkpoint loaded from {checkpoint_path}')
        return epoch + 1, losses, val_losses
    else:
        print(f'No checkpoint found at {checkpoint_path}. Starting from epoch 0')
        return 0, [], []

    # Load last checkpoint if exists, otherwise start from epoch 0


start_epoch, losses, val_losses = load_checkpoint(model, optimizer)

# Loop over epochs
for epoch in range(start_epoch, epochs):
    model.train()
    total_loss = 0.0

    print()
    for org_image, org_caption in train_loader:
        org_image = org_image.to(device)
        org_caption = org_caption.to(device)

        optimizer.zero_grad()
        outputs = model(org_image, org_caption[:, :-1])
        # print(outputs.reshape(-1, outputs.shape[2]).shape)
        # print(org_caption.reshape(-1).shape)

        org_caption = org_caption[:, 1:].reshape(-1)  # Ignore <sos> token
        loss = criterion(outputs.reshape(-1, vocabulary_size), org_caption)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Append epoch losses to the lists for plotting
    losses.append(total_loss / len(train_loader))

    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for org_image, org_caption in val_loader:
            org_image = org_image.to(device)
            org_caption = org_caption.to(device)

            outputs = model(org_image, org_caption[:, :-1])

            org_caption = org_caption[:, 1:].reshape(-1)
            loss = criterion(outputs.reshape(-1, vocabulary_size), org_caption)
            total_val_loss += loss.item()

    val_losses.append(total_val_loss / len(val_loader))

    print(
        f'Epoch [{epoch + 1}/{epochs}] | Training Loss = {total_loss / len(train_loader):.4f} | Validation Loss = {total_val_loss / len(val_loader):.4f}')

    # Save checkpoint after each epoch
    if val_losses[-1] <= min(val_losses):
        save_checkpoint(epoch, model, optimizer, losses, val_losses)
    # Plot encoder loss
    plt.plot(range(1, epoch + 2), losses, label='Training Loss', color='blue')
    plt.plot(range(1, epoch + 2), val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Captioner Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('loss_merge.png')
    plt.close()
