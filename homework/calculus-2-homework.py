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

# # Derivatives

# #### Example:
# $$f(x,y) = \begin{cases}x & \text{if }y \ne x^2 \\ 0 & \text{if }y = x^2\end{cases}$$

# ### proof:
# 

# #### Example:
# $$f(x,y) = \begin{cases}y^3/(x^2+y^2) & \text{if }(x,y) \ne (0,0) \\ 0 & \text{if }(x,y) = (0,0)\end{cases}$$
# 

# ### proof:
# We can find $f_x(0,0)$ and $f_y(0,0)$ <br>
# $f_x(0,0)=\lim_{h\to0} \frac{f(0+h,0)−f(0,0)}{h}=\lim_{h\to0} \frac{0}{h^3}=0$<br>
# $f_y(0,0)=\lim_{h\to0} \frac{f(0,0+h)−f(0,0)}{h}=\lim_{h\to0} \frac{h^3}{h^3}=1$<br>
# Both $f_x $ and $f_y$ exist at $(0,0)$, but they are not continuous at $(0,0)$, as
# $f_x=-2xy^3/(x^2+y^2)^-2$ and $f_y=(3 x^2 y^2 + y^4)/(x^2 + y^2)^2$ are not continuous at  $(0,0)$.<br>
# $\lim_{x,y\to(0,0)}y^3/(x^2+y^2)$ does not exist cause $\lim_{x\to 0}\lim_{y\to 0} y^3/(x^2+y^2)\neq \lim_{y\to 0}\lim_{x\to 0} y^3/(x^2+y^2)$ so $f$  is not continuous at $(0,0)$
# 

# 
