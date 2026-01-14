#!/usr/bin/env python
# coding: utf-8

# # Working with JAX numpy and calculating perplexity: Ungraded Lecture Notebook

# Normally you would import `numpy` and rename it as `np`. 
# 
# However in this week's assignment you will notice that this convention has been changed. 
# 
# Now standard `numpy` is not renamed and `trax.fastmath.numpy` is renamed as `np`. 
# 
# The rationale behind this change is that you will be using Trax's numpy (which is compatible with JAX) far more often. Trax's numpy supports most of the same functions as the regular numpy so the change won't be noticeable in most cases.
# 

# In[1]:


import numpy
import trax
import trax.fastmath.numpy as np

# Setting random seeds
trax.supervised.trainer_lib.init_random_number_generators(32)
numpy.random.seed(32)


# One important change to take into consideration is that the types of the resulting objects will be different depending on the version of numpy. With regular numpy you get `numpy.ndarray` but with Trax's numpy you will get `jax.interpreters.xla.DeviceArray`. These two types map to each other. So if you find some error logs mentioning DeviceArray type, don't worry about it, treat it like you would treat an ndarray and march ahead.
# 
# You can get a randomized numpy array by using the `numpy.random.random()` function.
# 
# This is one of the functionalities that Trax's numpy does not currently support in the same way as the regular numpy. 

# In[2]:


numpy_array = numpy.random.random((5,10))
print(f"The regular numpy array looks like this:\n\n {numpy_array}\n")
print(f"It is of type: {type(numpy_array)}")


# You can easily cast regular numpy arrays or lists into trax numpy arrays using the `trax.fastmath.numpy.array()` function:

# In[3]:


trax_numpy_array = np.array(numpy_array)
print(f"The trax numpy array looks like this:\n\n {trax_numpy_array}\n")
print(f"It is of type: {type(trax_numpy_array)}")


# Hope you now understand the differences (and similarities) between these two versions and numpy. **Great!**
# 
# The previous section was a quick look at Trax's numpy. However this notebook also aims to teach you how you can calculate the perplexity of a trained model.
# 

# ## Calculating Perplexity

# The perplexity is a metric that measures how well a probability model predicts a sample and it is commonly used to evaluate language models. It is defined as: 
# 
# $$P(W) = \sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{n-1})}}$$
# 
# As an implementation hack, you would usually take the log of that formula (to enable us to use the log probabilities we get as output of our `RNN`, convert exponents to products, and products into sums which makes computations less complicated and computationally more efficient). You should also take care of the padding, since you do not want to include the padding when calculating the perplexity (because we do not want to have a perplexity measure artificially good). The algebra behind this process is explained next:
# 
# 
# $$log P(W) = {log\big(\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{n-1})}}\big)}$$
# 
# $$ = {log\big({\prod_{i=1}^{N} \frac{1}{P(w_i| w_1,...,w_{n-1})}}\big)^{\frac{1}{N}}}$$ 
# 
# $$ = {log\big({\prod_{i=1}^{N}{P(w_i| w_1,...,w_{n-1})}}\big)^{-\frac{1}{N}}} $$
# $$ = -\frac{1}{N}{log\big({\prod_{i=1}^{N}{P(w_i| w_1,...,w_{n-1})}}\big)} $$
# $$ = -\frac{1}{N}{\big({\sum_{i=1}^{N}{logP(w_i| w_1,...,w_{n-1})}}\big)} $$

# You will be working with a real example from this week's assignment. The example is made up of:
#    - `predictions` : batch of tensors corresponding to lines of text predicted by the model.
#    - `targets` : batch of actual tensors corresponding to lines of text.

# In[4]:


from trax import layers as tl

# Load from .npy files
predictions = numpy.load('predictions.npy')
targets = numpy.load('targets.npy')

# Cast to jax.interpreters.xla.DeviceArray
predictions = np.array(predictions)
targets = np.array(targets)

# Print shapes
print(f'predictions has shape: {predictions.shape}')
print(f'targets has shape: {targets.shape}')


# Notice that the predictions have an extra dimension with the same length as the size of the vocabulary used.
# 
# Because of this you will need a way of reshaping `targets` to match this shape. For this you can use `trax.layers.one_hot()`.
# 
# Notice that `predictions.shape[-1]` will return the size of the last dimension of `predictions`.

# In[5]:


reshaped_targets = tl.one_hot(targets, predictions.shape[-1]) #trax's one_hot function takes the input as one_hot(x, n_categories, dtype=optional)
print(f'reshaped_targets has shape: {reshaped_targets.shape}')


# By calculating the product of the predictions and the reshaped targets and summing across the last dimension, the total log perplexity can be computed:

# In[6]:


total_log_ppx = np.sum(predictions * reshaped_targets, axis= -1)


# Now you will need to account for the padding so this metric is not artificially deflated (since a lower perplexity means a better model). For identifying which elements are padding and which are not, you can use `np.equal()` and get a tensor with `1s` in the positions of actual values and `0s` where there are paddings.

# In[7]:


non_pad = 1.0 - np.equal(targets, 0)
print(f'non_pad has shape: {non_pad.shape}\n')
print(f'non_pad looks like this: \n\n {non_pad}')


# By computing the product of the total log perplexity and the non_pad tensor we remove the effect of padding on the metric:

# In[8]:


real_log_ppx = total_log_ppx * non_pad
print(f'real perplexity still has shape: {real_log_ppx.shape}')


# You can check the effect of filtering out the padding by looking at the two log perplexity tensors:

# In[9]:


print(f'log perplexity tensor before filtering padding: \n\n {total_log_ppx}\n')
print(f'log perplexity tensor after filtering padding: \n\n {real_log_ppx}')


# To get a single average log perplexity across all the elements in the batch you can sum across both dimensions and divide by the number of elements. Notice that the result will be the negative of the real log perplexity of the model:

# In[10]:


log_ppx = np.sum(real_log_ppx) / np.sum(non_pad)
log_ppx = -log_ppx
print(f'The log perplexity and perplexity of the model are respectively: {log_ppx} and {np.exp(log_ppx)}')


# **Congratulations on finishing this lecture notebook!** Now you should have a clear understanding of how to work with Trax's numpy and how to compute the perplexity to evaluate your language models. **Keep it up!**

# In[ ]:




