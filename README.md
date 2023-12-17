# Shakespeare GPT
Tensorflow implementation of a generative pre-trained transformer based on Shakespeare's plays.

## Getting started
This project is built using Anaconda with the environment defined in environment.yaml. See the [Anaconda documentation](https://docs.anaconda.com/free/anaconda/install/index.html) for how to install Anaconda and set up the environment.

### Model training
The model is trained using the notebook `train.ipynb`. The notebook uses the `model.py` module to define the transformer model with the configuration specified in `config.py`.

## Sources
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin.
[Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
Advances in Neural Information Processing Systems 30, 2017.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever.
[Language Models are Unsupervised Multitask Learners](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe). 2019.

TensorFlow. 
[Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer). 
Tutorial.
