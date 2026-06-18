#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Basic setup

# Create anaconda environment
# <br>
# ```bash
# conda create -n ml python=3.7.4 jupyter
# ```
# Install fastai library
# <br>
# ```bash
# conda install -c pytorch -c fastai fastai
# ```

# # Set theory basics

# #### Prove that:
# <br>
# $A \subseteq A$

# ### proof:
# 
# For any elemnt $a\in A $ , also have $a  \in A $ so $A \subseteq A$
# 
# 

# #### Prove that:
# <br>
# If $A \subseteq B$ and $B \subseteq A$ $\to$ $A = B$

# ### proof:
# For every elemnt $a\in A $ , also have $a  \in B $ and for every elemnt $b\in B$ , also have $b \in A \longrightarrow$ $A = B$
# 

# #### Prove that:
# <br>
# if $B \subset A$ then $A \cap B = B$

# ### proof:
# By deffinition :Intersection of two sets $A$ and $B$ $A \cap B$ is a "biggest" subset of $A$ and $B$, that means for every set $C$ such that, $C \subseteq A$ and $C \subseteq B$ $\to$  $C \subseteq A\cap B$ <br>
# $B \subset A\longrightarrow$ for every elemnt $a\in B$ , also have $a \in A    \longrightarrow B\subseteq A\cap B $
# and biggest" subset of B is B so $A \cap B = B$

# #### Prove that:
# <br>
# $A \cap B = B \cap A$

# ### proof:
# For any $a\in A \cap B , a\in A$ and $a\in B$  <br>if  $a\in B$ and $a\in A$ $\longrightarrow a\in B \cap A$ and vice versal so  $A \cap B = B \cap A$

# #### Prove that:
# <br>
# if $B \subset A$ then $A \cup B = A$

# ### proof:
# By definniton  Union of two sets $A$ and $B$ $A \cup B$ is a "smallest" set that contains both - $A$ and $B$, that means:
# for every set $D$ such that $A \subseteq D$ and $B \subseteq D$ $\to$ $A \cup B \subseteq D$ <br>
# The smallest set wich contains $A$ is $A$... and  $B \subset A\longrightarrow$  $A$ contains $B$ as well so $A \cup B = A$
# 

# #### Prove that:
# <br>
# $A \cup B = B \cup A$

# ### proof:
# By definniton  Union of if $A \cup B=C\longrightarrow C$ Is the smallest set wich contains $A$ and $B$ sets $\longrightarrow C$ is smallest set wich contains $B$ and $A\longrightarrow C=B \cup A\longrightarrow A \cup B = B \cup A$ 

# #### Prove that:
# - for every injection $m:A \to B$ and pair of functions $f, g :C \to A$: if $m \circ f = m \circ g$ then $f = g$ and vice-versa
# - for every surjection $e:A \to B$ and every pair of functions $f, g :B \to C$: if $f \circ e = g \circ e$ then $f = g$ and vice-versa

# #### Prove that 
# - composition of injections is injection itself
# - composition of surjections is surjection itself
# - composition of bijections is bijection itself
# <br>
# or give a counterexamples
# 

# #### Prove that for each set $A$:
# - $A \cong A$
# - if $B \cong A$ then $B \cong A$ for every pair of sets $A$ and $B$
# - if $A \cong B$ and $B \cong C$ then $A \cong C$ for every triplet $A$, $B$ and $C$

# #### Prove that:
# <br>
# there exists a bijection between set of natural and even numbers

# #### Prove that:
# <br>
# if we have a bijection between two finite sets than they have an equal number of elements

# #### Prove that:
# <br>
# $A \times B \cong B \times A$

# $\cap_{i\in I}A_i$ and $\cup_{i\in I}A_i$

# In[ ]:


# Inplement in python


# We can also define cartesian product of any "number" of sets $\prod_{i \in I}{A_i}$

# In[ ]:


# Inplement in python


# #### Prove that:
# <br>
# $$A \cap (B \cup C)=(A \cap B) \cup (A\cap C)$$
# $$A \cup (B \cap C)=(A \cup B) \cap (A\cup C)$$

# # Linear Algebra

# #### Prove that:
# <br>
# $(AB)^{T} = B^{T}A^{T}$ for each pair of matrices $A, B \in \mathbb{R}^{n \times m}$
# ### Proof:
# $(AB)^{T} = B^{T}A^{T}$ we can multiply both sides to $AB$ 

# ## Functions on tensors

# #### Write combination for $XOR$ calculation

# In[ ]:




