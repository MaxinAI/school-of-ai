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

# In[ ]:


get_ipython().system('pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master')
get_ipython().system('pip install jupyter_nbextensions_configurator')
get_ipython().system('jupyter contrib nbextension install --user')
get_ipython().system('jupyter nbextensions_configurator enable --user')


# # Metric spaces

# #### Prove that:
# For every metric space (X, d):
# - For eny $\mathcal{U} = \{U | U$ is open in $(X, d)\}$ holds $\bigcup_{U \in \mathcal{U}} U$ is open in $(X, d)$
# - For eny two $U, V \subset X$ open in $(X, d)$ holds: $U \cap V$ is open in $(X, d)$
# - $X$ is open in $(X, d)$
# - $\emptyset$ is open in $(X, d)$

# #### Prove that:
# - For eny finity set $(U)_{i=1}^{n}$ of open sets in $(X, d)$, $\bigcap_{i=1}^{n}U$ is open in $(X, d)$

# #### Prove that for set $U \subset X$ is open if and only if for each point $x \in U$ there exists the open neighbourhood $V$ of $x$ such that $V \subset U$

# #### Prove that, closed ball is closed subset in $(X, d)$
# 

# #### Prove that
# For every metric space (X, d):
# - For eny $\mathcal{F} = \{F | F$ is closed in $(X, d)\}$ holds $\bigcap_{F \in \mathcal{F}} F$ is closed in $(X, d)$
# - For eny two $F_1, F_2 \subset X$ closed in $(X, d)$ holds: $F_1 \cup F_2$ is closed in $(X, d)$
# - $X$ is closed in $(X, d)$
# - $\emptyset$ is closed in $(X, d)$

# #### Prove that:
# - For eny finity set $(F)_{i=1}^{n}$ of closed sets in $(X, d)$, $\bigcup_{i=1}^{n}F$ is closed in $(X, d)$

# #### Prove that, if $F \subset X$ is closed then $X - F$ is open in $(X, d)$ 

# # Metrics in Euclidean spaces

# #### Prove that:
# - for every $u, v \in \mathbb{R}^{n}$: $d(u, v) \geq 0$
# - for every $v \in \mathbb{R}^{n}$: $d(v, v) = 0$
# - for every $u, v \in \mathbb{R}^{n}$: $d(u, v) = d(v, u)$ (symmetry)
# - for every $u, v, w \in \mathbb{R}^{n}$: $d(u, w) \leq d(v, u) + d(v, w)$ (triangle inequality)

# #### Prove the same properties hold for $d(u, v) = ||u-v||_1$ ($||u-v||_1 = \sum_{i = 1}^{n}|u_i - v_i|$)

# ## Sequences and limits

# #### Prove that $x = \lim_{n\to\infty}{x_n}$ in $(X, d)$ if and only if (iff) for every $r \in \mathbb{R}$ there exists $n_0 \in \mathbb{N}$ such that: $x_i \in B(x, r)$ for every $i \gt n_0$

# #### Prove that if $x = \lim_{n\to\infty}{x_n}$ and $x \notin \{-\infty, \infty\}$ then $(x_i)_{i=1}^{\infty} = (x_1, x_2, \dots, x_n)$ is a Cauchy sequence
# <br>
# For closed set $F \subset \mathbb{R}^n$ and convergent sequence $(x_i)_{i=1}^{\infty} = (x_1, x_2, \dots, x_n)$ such that there exists $n_0 \in \mathbb{N}$ such that $x_i \in F$ for each $i \gt n_0$ then: $\lim_{n\to\infty}{x_n} \in F$

# #### Prove that if $F$ is open from previous example, statement does not hold. 

# #### Prove that inherited metric is a metric

# ## Limits of functions
# 

# Let $f:S \to Y$ is function between subset $S \subset X$ of a metric space $(X, d_x)$ and metric space $(Y, d_Y)$
# - We say that the limit of function $f:(S, d_x) \to (Y, d_Y)$ between metric spaces in some limit point $c \in X$ of the subset $S$ is $y \in Y $if for each open neighborhood of $y \in V \subset Y$ there exists the open seighborhood of $c \in U \subset X$ such that $f(U \cap S) \subset V$
# <br>
# This definition is equiualent of definition:
# <br>
# - The limit of function $f:(S, d_X) \to (Y, d_Y)$ between metric spaces in limit point $c \in X$ of the subset $S$ is $y \in Y $ if for each open ball $B(x, r) \subset Y$ there exists the ball $B(c, l) \subset X$ such that $f(B(c, l) \cap S) \subset B(y, r)$
# <br>
# or
# <br>
# - The limit of function $f:(S, d_X) \to (Y, d_Y)$ between metric spaces in limit point $c \in X$ of subset $S$ is $y \in Y $ if for any $r \in \mathbb{R}$ there exists $l \in \mathbb{R}$ such that for every $x \in S$ with $d_X(x, c) < l$ implies that $d_Y(f(x), y) < r$

# #### Prove that this three definitions are equiualent for eny function between eny two metric spaces

# ## Continuous functions 

# #### Prove that function is continuous in $c$ if for eny sequence $(x_n)_{n=1}^{\infty} \subset X$ such that $\lim_{n \to \infty}x_n = c$ we have $\lim_{n \to \infty}f(x_n) = f(c)$

# ## Proof 
# 
# $\any$
#     
# 

# #### Prove that function is continuous if for every open set $V \subset Y$ the $f^{-1}(V)$ is open in $X$
# 
#     
#     

# #### Prove that function is continuous if for every closed set $F \subset Y$ the $f^{-1}(F)$ is open in $X$

# #### Prove that any composition of continous functions is continous

# In[ ]:




