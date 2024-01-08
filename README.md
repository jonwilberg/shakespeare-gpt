# Shakespeare GPT
Tensorflow implementation of a generative pre-trained transformer based on Shakespeare's plays.

### Examples of generated output
Prompt: 
```
JULIET : 
O Romeo , Romeo ! wherefore art thou Romeo ?
```

Generated output:
```
No , faith I love thee , boy , but in loving soul ,
My love as sovereign as myself , fair Montague .
Thy honour is thy name , and thy meritate :
For every thing thou dost me know , do me wrong ;
What that , makest fruit weep , yea , from thy heart ,
In thy nails , the father of thy love ,
Answer the kings in love , the tables for her :
And if he be a rule , then the a
```

Prompt: 
```
HAMLET :
To be , or not to be : that is the question :
```

Generated output:
```
to speak on God , I was but read ; and your shape is
too single for you , and pray for that .

HORATIO :
I am like dark as this is the bidding .

MARCELLUS :
My Lord of Spurrey , go to the king with him .

BERNARDO :
What , shall we part ?

HORATIO :
' Tis private , my lord ;
For this same hour will come and hold our duty ,
Which neither by imaginer hath 
```

## Getting started
This project is built using Anaconda with the environment defined in environment.yaml. See the [Anaconda documentation](https://docs.anaconda.com/free/anaconda/install/index.html) for how to install Anaconda and set up the environment.

### Model training
The model is trained using the script `train.py`. The notebook uses the `model.py` module to define the transformer model with the configuration specified in `config.py`. The dataset is preprocessed and the tokenizer is trained in the `tokenization.ipynb` notebook.

## Sources
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser and Illia Polosukhin.
[Attention is All you Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).
Advances in Neural Information Processing Systems 30, 2017.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever.
[Language Models are Unsupervised Multitask Learners](https://www.semanticscholar.org/paper/Language-Models-are-Unsupervised-Multitask-Learners-Radford-Wu/9405cc0d6169988371b2755e573cc28650d14dfe). 2019.

TensorFlow. 
[Neural machine translation with a Transformer and Keras](https://www.tensorflow.org/text/tutorials/transformer). 
Tutorial.

Andrej Karpathy.
[Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).
YouTube video.
