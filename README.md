# PyTorch Triplet Loss and Online Mining
- Colab Notebook: https://colab.research.google.com/drive/1_1jwzyqdJWxoMgc7nKh-kPszJyxJ0xr3?usp=sharing
- Implementation of triplet loss, and online mining on Pytorch for training on for the EMNIST dataset.

# Model:
- Since this model is designed for a different dataset from the original, I have implemented a modified version of Resnet18 architecture (tweaks to strides, and the final linear sequential), instead of the more rudimentary one in the original repo. (which is for MNIST)
- Embedding size have also been modified.
- All model training experiments documented here: https://docs.google.com/spreadsheets/d/1_O3CetFmNNdVCo2OIOdrkai5emCOmzQpxxYBG5hFcp4/edit?usp=sharing

# Task:
## Training:
- This code randomly picks 20 classes from the EMNIST dataset and trains the network using Triplet loss. (Feature vector) (I have used the 'hard' config to force harder examples and get better results)
- After training, you can visualize the new clusters.
## Testing:
- As per the assignment, the goal is to check how well the model performs as a classifier for classes it hasn't seen before. 
- Hence, we pick the remaining 6 classes, create threshold limits for classification based on the intra-class distances of the embeddings (derived from inference) and use threshold value to classify whether 2 pairs of images are from the same class or not.
- I managed to achieve an accuracy of 82.5% after 100 iterations on the Resnet model.

## Future Work:
- Working on 'hybrid_resnet' model that increases the level of customization to the network and filters in order to optimize learning.



# Dependencies
pytorch >=1.3

sklearn >=0.19.1

matplotlib >=2.2.2

seaborn >=0.9

# Instruction
All the required codes are contained inside the jupyter-notebook: (https://colab.research.google.com/drive/1_1jwzyqdJWxoMgc7nKh-kPszJyxJ0xr3?usp=sharing)

