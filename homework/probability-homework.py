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

# # Sigma-algebras

# #### Prove that:
# Composition of mesurable functions is a mesurable function

# ## Mesure

# #### Prove the following mesure properties:
# - if $A \subset B$ then $\mu(A) \le \mu(B)$
# <br>
# Proof: $(B - A) \cup (A \cap B) = (B - A) \cup A = B$ and $(B - A) \cap (A \cap B) = \emptyset$ so $\mu((B - A) \cup (A \cap B)) = \mu(B - A) + \mu(A \cap B) = \mu(B - A) + \mu(A) = \mu(B)$ so $\mu(A) \le \mu(B)$

# ## Probability

# #### Prove that:
# For probability mesure $P$ and events $A, B, C \subset \Omega$
# - $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
# <br>
# 
# ### proof
# #### lema 1. 
# $B=(A\cap B)\cup (B-A)$, also  $(A\cap B)$ and $(B-A)$ are disjoined sets $\implies$ $P(B)=P(A\cap B)+P(B-A)\implies P(B-A)=P(B)-P(A\cap B)$ <br>
# #### ***
# $(A\cup B)=A\cup (B-A)$, aslo $A$ and $(B-A)$ are disjoint sets $\implies P(A\cup B)=P(A)+P(B-A)$ and according the lema 1 $P(B-A)=P(B)-P(A\cap B)$ so we can say that   $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
# 
# 

# ## Random variable

# #### Prove that:
# $$
# \sigma^2 = \operatorname{E}[X^2] - (\operatorname{E}[X])^2
# $$
# Assuming that expectation exists
# <br>
# Denoted by $\sigma^2$
# <br>

# #### Proof:
# We know that $\sigma^2 = \operatorname{Var}(X)=  \operatorname{E}[(X - \operatorname{E}[X])^2] = \operatorname{E}[X^2 -2X\operatorname{E}[X] + (\operatorname{E}[X])^2]= \operatorname{E}[X^2] -\operatorname{E}[2X\operatorname{E}[X]] + \operatorname{E}[(\operatorname{E}[X])^2]$ <br>
# Noticed that $\operatorname{E}[X] $ is a number so $\operatorname{E}[2X\operatorname{E}[X]] = 2\operatorname{E}[X]\operatorname{E}[X]=2(\operatorname{E}[X])^2$ and $\operatorname{E}[(\operatorname{E}[X])^2]=(\operatorname{E}[X])^2\Longrightarrow $ $\sigma^2=\operatorname{E}[X^2]-2(\operatorname{E}[X])^2+(\operatorname{E}[X])^2=\operatorname{E}[X^2]-(\operatorname{E}[X])^2$

# 
