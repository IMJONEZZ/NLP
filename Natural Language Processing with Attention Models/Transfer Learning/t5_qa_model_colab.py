# -*- coding: utf-8 -*-
"""T5 QA Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lkqIfpzEB2Apybo_NSJcPFZmIXhdQMkM

# Assignment 3, Part 2: T5 SQuAD Model 

Welcome to the part 2 of testing the models for this week's assignment. This time we will perform decoding using the T5 SQuAD model. In this notebook we'll perform Question Answering by providing a "Question", its "Context" and see how well we get the "Target" answer. 

## IMPORTANT

- As you cannot save the changes you make to this colab, you have to make a copy of this notebook in your own drive and run that. You can do so by going to `File -> Save a copy in Drive`. Close this colab and open the copy which you have made in your own drive.

- Go to this [google drive folder](https://drive.google.com/drive/folders/1rOZsbEzcpMRVvgrRULRh1JPFpkIG_JOz?usp=sharing) named `NLP C4 W3 Data`. In the folder, next to its name use the drop down menu to select `"Add shortcut to Drive" -> "My Drive" and then press ADD SHORTCUT`. This should add a shortcut to the folder `NLP C4 W3 Data` within your own google drive. Please make sure this happens, as you'll be reading the data for this notebook from this folder.

- Make sure your runtime is GPU (_not_ CPU or TPU). And if it is an option, make sure you are using _Python 3_. You can select these settings by going to `Runtime -> Change runtime type -> Select the above mentioned settings and then press SAVE`

**Note: Restarting the runtime maybe required**.

Colab will tell you if the restarting is necessary -- you can do this from the:

Runtime > Restart Runtime

option in the dropdown.

## Outline

- [Part 0: Downloading and loading dependencies](#0)
- [Part 1: Mounting your drive for data accessibility](#1)
- [Part 2: Getting things ready](#2)
- [Part 3: Fine-tuning on SQuAD](#3)
    - [3.1 Loading in the data and preprocessing](#3.1)
    - [3.2 Decoding from a fine-tuned model](#3.2)

<a name='0'></a>
# Part 0: Downloading and loading dependencies

Uncomment the code cell below and run it to download some dependencies that you will need. You need to download them once every time you open the colab. You can ignore the `kfac` error.
"""

!pip -q install trax

import string
import t5
import numpy as np
import trax 
from trax.supervised import decoding
import textwrap 
# Will come handy later.
wrapper = textwrap.TextWrapper(width=70)

"""<a name='1'></a>
# Part 1: Mounting your drive for data accessibility

Run the code cell below and follow the instructions to mount your drive. The data is the same as used in the coursera version of the assignment.
"""

from google.colab import drive
drive.mount('/content/drive/', force_remount=True)

"""<a name='2'></a>
# Part 2: Getting things ready 

Run the code cell below to ready some functions which will later help us in decoding. The code and the functions are the same as the ones you previsouly ran on the coursera version of the assignment.
"""

PAD, EOS, UNK = 0, 1, 2

def detokenize(np_array):
  return trax.data.detokenize(
      np_array,
      vocab_type='sentencepiece',
      vocab_file='sentencepiece.model',
      vocab_dir='/content/drive/My Drive/NLP C4 W3 Data/')

def tokenize(s):
  # The trax.data.tokenize function operates on streams,
  # that's why we have to create 1-element stream with iter
  # and later retrieve the result with next.
  return next(trax.data.tokenize(
      iter([s]),
      vocab_type='sentencepiece',
      vocab_file='sentencepiece.model',
      vocab_dir='/content/drive/My Drive/NLP C4 W3 Data/'))
 
vocab_size = trax.data.vocab_size(
    vocab_type='sentencepiece',
    vocab_file='sentencepiece.model',
    vocab_dir='/content/drive/My Drive/NLP C4 W3 Data/')

def get_sentinels(vocab_size):
    sentinels = {}

    for i, char in enumerate(reversed(string.ascii_letters), 1):

        decoded_text = detokenize([vocab_size - i]) 
        
        # Sentinels, ex: <Z> - <a>
        sentinels[decoded_text] = f'<{char}>'
        
    return sentinels

sentinels = get_sentinels(vocab_size)    


def pretty_decode(encoded_str_list, sentinels=sentinels):
    # If already a string, just do the replacements.
    if isinstance(encoded_str_list, (str, bytes)):
        for token, char in sentinels.items():
            encoded_str_list = encoded_str_list.replace(token, char)
        return encoded_str_list
  
    # We need to decode and then prettyfy it.
    return pretty_decode(detokenize(encoded_str_list))

"""<a name='3'></a>
# Part 3: Fine-tuning on SQuAD

Now let's try to fine tune on SQuAD and see what becomes of the model. For this, we need to write a function that will create and process the SQuAD `tf.data.Dataset`. Below is how T5 pre-processes SQuAD dataset as a text2text example. Before we jump in, we will have to first load in the data. 

<a name='3.1'></a>
### 3.1 Loading in the data and preprocessing

You first start by loading in the dataset. The text2text example for a SQuAD example looks like:

```
{
  'inputs': 'question: <question> context: <article>',
  'targets': '<answer_0>',
}
```

The squad pre-processing function takes in the dataset and processes it using the sentencePiece vocabulary you have seen above. It generates the features from the vocab and encodes the string features. It takes on question, context, and answer, and returns "question: Q context: C" as input and "A" as target.
"""

# Retrieve Question, C, A and return "question: Q context: C" as input and "A" as target.
def squad_preprocess_fn(dataset, mode='train'):
  return t5.data.preprocessors.squad(dataset)

# train generator, this takes about 1 minute
train_generator_fn, eval_generator_fn = trax.data.tf_inputs.data_streams(
  'squad/v1.1:2.0.0',
  data_dir='/content/drive/My Drive/NLP C4 W3 Data/data/',
  bare_preprocess_fn=squad_preprocess_fn,
  input_name='inputs',
  target_name='targets'
)
train_generator = train_generator_fn()
next(train_generator)

#print example from train_generator
(inp, out) = next(train_generator)
print(inp.decode('utf8').split('context:')[0])
print()
print('context:', inp.decode('utf8').split('context:')[1])
print()
print('target:', out.decode('utf8'))

"""<a name='3.2'></a>
### 3.2 Decoding from a fine-tuned model

You will now use an existing model that we trained for you. You will initialize, then load in your model, and then try with your own input.
"""

# Initialize the model 
model = trax.models.Transformer(
    d_ff = 4096,
    d_model = 1024,
    max_len = 2048,
    n_heads = 16,
    dropout = 0.1,
    input_vocab_size = 32000,
    n_encoder_layers = 24,
    n_decoder_layers = 24,
    mode='predict')  # Change to 'eval' for slow decoding.

# load in the model
# this will take a minute
shape11 = trax.shapes.ShapeDtype((1, 1), dtype=np.int32)
model.init_from_file('/content/drive/My Drive/NLP C4 W3 Data/models/model_squad.pkl.gz',
                     weights_only=True, input_signature=(shape11, shape11))

# Uncomment to see the transformer's structure.
# print(model)

# create inputs
# a simple example 
# inputs = 'question: She asked him where is john? context: John was at the game'

# an extensive example
inputs = 'question: What are some of the colours of a rose? context: A rose is a woody perennial flowering plant of the genus Rosa, in the family Rosaceae, or the flower it bears.There are over three hundred species and tens of thousands of cultivars. They form a group of plants that can be erect shrubs, climbing, or trailing, with stems that are often armed with sharp prickles. Flowers vary in size and shape and are usually large and showy, in colours ranging from white through yellows and reds. Most species are native to Asia, with smaller numbers native to Europe, North America, and northwestern Africa. Species, cultivars and hybrids are all widely grown for their beauty and often are fragrant.'

# tokenizing the input so we could feed it for decoding
print(tokenize(inputs))
test_inputs = tokenize(inputs)

# Temperature is a parameter for sampling.
#   # * 0.0: same as argmax, always pick the most probable token
#   # * 1.0: sampling from the distribution (can sometimes say random things)
#   # * values inbetween can trade off diversity and quality, try it out!
output = decoding.autoregressive_sample(model, inputs=np.array(test_inputs)[None, :],
                                        temperature=0.0, max_length=10)
print(wrapper.fill(pretty_decode(output[0])))

"""### Note: As you can see the RAM is almost full, it is because the model and the decoding is memory heavy. You can run decoding just once. Running it the second time with another example might give you the same answer as before, or not run at all (crash). If that happens restart the runtime (see how to at the start of the notebook) and run all the cells again."""