import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from GANModels import ImageEmbedding, TextEmbedding
from DataLoaders import train_loader, test_loader, val_loader, vocabulary_size, caption_vocab

# Hyperparameters
lr = 3e-4
image_size = 224
batch_size = 32
epochs = 10
embed_size = 256
text_embed_size = 50
hidden_size = 128
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
checkpoint_dir = 'gan_checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)

# Models
generator = ImageEmbedding(embed_size, max_caption_length).to(device)
discriminator = TextEmbedding(embed_size, vocabulary_size, hidden_size, max_caption_length).to(device)

# Optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr)
disc_optimizer = optim.Adam(discriminator.parameters(), lr)


# Loss Functions
def gen_loss(outputs):
    return torch.sum(torch.log(1 - outputs + 1e-6))


def disc_loss(real, fakes):
    return -1 * torch.sum(torch.log(real + 1e-6) + torch.log(1 - fakes + 1e-6))


# Function to save checkpoint
def save_checkpoint(epoch, generator, discriminator, gen_optimizer, disc_optimizer, gen_losses, disc_losses):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch.pt')
    torch.save({
        'epoch': epoch,
        'genLoss': gen_losses,
        'discLoss': disc_losses,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'gen_optimizer_state_dict': gen_optimizer.state_dict(),
        'disc_optimizer_state_dict': disc_optimizer.state_dict()
    }, checkpoint_path)
    print(f'Checkpoint saved at {checkpoint_path}')


# Function to load checkpoint
def load_checkpoint(generator, discriminator, gen_optimizer,  disc_optimizer):
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        epoch = checkpoint['epoch']
        gen_losses = checkpoint['genLoss']
        disc_losses = checkpoint['discLoss']
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        print(f'Checkpoint loaded from {checkpoint_path}')
        return epoch + 1, gen_losses, disc_losses
    else:
        print(f'No checkpoint found at {checkpoint_path}. Starting from epoch 0')
        return 0, [], []

    # Load last checkpoint if exists, otherwise start from epoch 0


start_epoch, gen_losses, disc_losses = load_checkpoint(generator, discriminator, gen_optimizer, disc_optimizer)

# Loop over epochs
for epoch in range(start_epoch, epochs):
    generator.train()
    discriminator.train()
    total_gen_loss = 0.0
    total_disc_loss = 0.0

    for org_image, org_caption in train_loader:
        org_image = org_image.to(device)
        org_caption = org_caption.to(device)

        fake_captions = generator(org_image)

        # Discriminator Training
        fakes = discriminator(fake_captions.argmax(2))
        reals = discriminator(org_caption)

        disc_optimizer.zero_grad()
        loss_disc = disc_loss(reals, fakes)
        loss_disc.backward(retain_graph=True)
        disc_optimizer.step()

        # Training Generator
        outputs = discriminator(fake_captions.argmax(2))
        gen_optimizer.zero_grad()
        loss_gen = gen_loss(outputs)
        loss_gen.backward()
        gen_optimizer.step()

        total_gen_loss += loss_gen.item()
        total_disc_loss += loss_disc.item()

    print(f'Epoch [{epoch + 1}/{epochs}] | Generator Loss = {total_gen_loss / len(train_loader):.4f} | Discriminator Loss = {total_disc_loss / len(train_loader):.4f}')

    # Append epoch losses to the lists for plotting
    gen_losses.append(total_gen_loss / len(train_loader))
    disc_losses.append(total_disc_loss / len(train_loader))

    # Save checkpoint after each epoch
    save_checkpoint(epoch, generator, discriminator, gen_optimizer, disc_optimizer, gen_losses, disc_losses)

    generator.eval()
    discriminator.eval()
    batch_images, batch_tensors = next(iter(test_loader))
    batch_images = batch_images.to(device)
    batch_tensors = batch_tensors.to(device)

    preds = generator(batch_images).argmax(2)

    print("Real Sentence =", get_real_captions(reversed_vocab, batch_tensors)[:3])
    print("Predicted Sentence =", get_real_captions(reversed_vocab, preds)[:3])

    plt.figure(figsize=(15, 5))

    # Plot encoder loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epoch + 2), gen_losses, label='Encoder Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Generator Loss Curve')
    plt.legend()
    plt.grid(True)

    # Plot decoder loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epoch + 2), disc_losses, label='Decoder Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('gan_loss.png')
    plt.close()
