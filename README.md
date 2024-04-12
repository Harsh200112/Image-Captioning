
# Image Captioning

Image captioning for the visually impaired is a significant issue that needs attention. It involves generating descriptions for images understandably and accurately. This task is crucial to aid visually impaired individuals in interpreting and understanding visual content. Despite advancements in technology, creating precise and contextually appropriate captions remains a challenging task. Our project aims to address this problem and develop an efficient solution that can generate accurate and meaningful captions for images, thereby enhancing the experience for visually impaired individuals.

The main focus of this project is to generate an interpretable and meaningful set of captions for real-life images. We have also converted the generated caption to audio for the visually impaired to listen to the generated captions.

There are 6 Folders and 1 pdf Report files in this zipped folder:-
- General Architecture
- GAN Architecture
- VAE Architecture
- Merge Architecture
- Proposed Model With CNN Architecture
- Proposed Model with ResNet Architecture
- Report.pdf

Colab noteboook files are available for the following folders General Architecture, VAE Architecture, Proposed Model With CNN Architecture and Proposed Model with ResNet Architecture for which you have to simply upload the those files change the path of the dataset in the file and run the file to train the model and get inferences. The link to the colab files have also been given in the report itself.
## Steps to run GAN and Merge Architecture

In order to train GAN Architecture on your device open the folder of GAN Architecture and open command prompt in it. After which run the following command
```
python TrainGAN.py
```

In order to train Merge Architecture on your device open the folder of Merge Architecture and open command prompt in it. After which run the following command
```
python Train.py
```
In the case of Merge Architecture you will also have to provide the location to the weights of the pretrained resnet152 model.

Note:- Here again you will have to sepcify the location of Dataset in the DataLoaders.py file of the respective Architectures.

Result Folder in the Merge Architecture also has the results which we got after traning the model.
## Acknowledgements

 - [General Architecture (Google's Architecture)](https://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
 - [A neural image caption generator.](https://arxiv.org/pdf/1411.4555.pdf)
 - [Practical Implementation Video](https://www.youtube.com/watch?v=y2BaTt1fxJU&list=PLCJHEFznK8ZybO3cpfWf4gKbyS5VZgppW)
 - [Dense Contrastive Learning for Self-Supervised Visual Pre-Training. ](https://arxiv.org/pdf/2103.00020.pdf)

