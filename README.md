# English to Norwegian Transformer
Tensorflow implementation of a transformer for English to Norwegian translation. The transformer is trained on the [ParaCrawl Corpus release v9](https://paracrawl.eu/releases).

## Getting started
This project is built using Anaconda with the environment defined in environment.yaml. See the [Anaconda documentation](https://docs.anaconda.com/free/anaconda/install/index.html) for how to install Anaconda and set up the environment.

### Data preparation
The Norwegian (Bokmål) ParaCrawl Corpus is downloaded from [here](https://paracrawl.eu/releases). The corpus is filtered, split into training and validation sets, and tokenized using the script in the notebook `tokenization.ipynb`. The resulting files are saved in the `data` folder.

### Model training
The model is trained using the notebook `train.ipynb`. The notebook uses the `model.py` module to define the transformer model with the configuration specified in `config.py`.

## Sources
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin.
[Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
Advances in Neural Information Processing Systems 30, 2017.

Peter Bloem.
[Transformers from scratch](https://peterbloem.nl/blog/transformers).
Blog post, 2019.

Jakob Uszkoreit,. 
[Transformer: A Novel Neural Network Architecture for Language Understanding](https://blog.research.google/2017/08/transformer-novel-neural-network.html). 
Blog post, 2017.

TensorFlow. 
[Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer). 
Tutorial.

Marta Bañón, Pinzhen Chen, Barry Haddow, Kenneth Heafield, Hieu Hoang, Miquel Esplà-Gomis, Mikel L. Forcada, Amir Kamran, Faheem Kirefu, Philipp Koehn, Sergio Ortiz Rojas, Leopoldo Pla Sempere, Gema Ramírez-Sánchez, Elsa Sarrías, Marek Strelec, Brian Thompson, William Waites, Dion Wiggins, and Jaume Zaragoza. [ParaCrawl: Web-Scale Acquisition of Parallel Corpora](https://aclanthology.org/2020.acl-main.417/). In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020.
