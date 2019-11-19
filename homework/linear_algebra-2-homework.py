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

# # Linear spaces

# #### Prove that:
# For every linear function $f:V \to U$ between vector spaces $V$ and $U$ on the real numbers, every sequence of vectors $v_1, v_2, \dots v_m \in V$ and every scalars $a_1, a_2, \dots, a_m \in \mathbb{R}$:
# <br>
# $$f(a_1v_1 + a_2v_2 + \dots + a_mv_m) = a_1f(v_1) + a_2f(v_2) + \dots + a_mf(v_m)$$

# #### Proof:
# $f:V \to U$ between vector spaces $\mathbb{U}$ and $\mathbb{V}$ is called linear (or linear transformation) if for every $u, v \in \mathbb(U)$ and every scalar $\alpha  \in \mathbb{R}^{1}$ we have:
# - $f(u + v) = f(u) + f(v)$
# - $f(\alpha u) = \alpha f(u)$
# <br></br>
# $f(u + v) = f(u) + f(v)\longrightarrow f(a_1v_1 + a_2v_2 + \dots + a_mv_m) = f(a_1v_1) + f(a_2v_2) + \dots + f(a_mv_m)$<br>
# $f(\alpha u) = \alpha f(u)\longrightarrow f(a_1v_1) + f(a_2v_2) + \dots + f(a_mv_m)=a_1f(v_1) + a_2f(v_2) + \dots + a_mf(v_m)$ 
# <br>
# So we can say that $f(a_1v_1 + a_2v_2 + \dots + a_mv_m) = a_1f(v_1) + a_2f(v_2) + \dots + a_mf(v_m)$

# In[ ]:




