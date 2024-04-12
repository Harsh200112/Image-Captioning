import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from Models import ImageCaptioner
from DataLoaders import test_loader, vocabulary_size, caption_vocab

# Hyperparameters
lr = 3e-4
image_size = 224
batch_size = 32
epochs = 50
embed_size = 512
text_embed_size = 256
hidden_size = 256
image_hid_size = 1024
max_caption_length = 28
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
reversed_vocab = {value: key for key, value in caption_vocab.items()}


def get_real_captions(reversed_vocab, batch_tensors):
    sentences = []
    for tensor in batch_tensors:
        words = []
        for number in tensor:
            if reversed_vocab[int(number)] != '<eos>':
                words.append(reversed_vocab[int(number)])
            else:
                break

        sentence = ' '.join(filter(None, words[1:]))
        sentences.append(sentence)

    return sentences


def display_image_with_captions(image, real_caption, generated_caption):
    image = image.cpu().squeeze(0)
    image = transforms.ToPILImage()(image)

    plt.imshow(image)
    plt.axis('off')

    # Add real and generated captions below the image
    plt.text(0, image.size[1], "Real Caption: " + real_caption, fontsize=12, wrap=True, verticalalignment='top')
    plt.text(0, image.size[1] + 20, "Generated Caption: " + generated_caption, fontsize=12, wrap=True,
             verticalalignment='top')

    plt.show()


def pad_sequence(sequence, max_length, vocab):
    if len(sequence) < max_length:
        sequence += [vocab['<pad>']] * (max_length - len(sequence))

    return sequence


def generate_caption(model, image, max_length=20):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        # Initialize the caption with start token
        caption = [caption_vocab['<sos>']]
        for _ in range(max_length):
            # Convert the current caption to tensor
            caption_tensor = torch.tensor([caption]).to(device)
            # Generate output for the current caption
            output = model(image, caption_tensor).argmax(2)
            # Get the last word predicted
            last_word = output[0][-1].item()
            # Append the word to the caption
            caption.append(last_word)
            # If end token is predicted, stop generating
            if last_word == caption_vocab['<eos>']:
                break
    # Convert the caption from indices to words
    generated_caption = ' '.join([reversed_vocab[word] for word in caption[1:-1]])
    return generated_caption


model = ImageCaptioner(embed_size, text_embed_size, vocabulary_size, hidden_size, image_hid_size).to(device)

checkpoint_path = 'merge_checkpoints/checkpoint_epoch.pt'
checkpoint = torch.load(checkpoint_path)

model.load_state_dict(checkpoint['model_state_dict'])

# Choose a single image and caption from the test_loader
image, caption = next(iter(test_loader))
image = image[0].unsqueeze(0).to(device)  # Choose the first image from the batch
caption = caption[0].unsqueeze(0).to(device)  # Choose the corresponding caption

real_caption = get_real_captions(reversed_vocab, caption)
# Generate caption for the chosen image
generated_caption = generate_caption(model, image)

display_image_with_captions(image, real_caption[0], generated_caption)
