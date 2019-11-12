conda install nb_conda
pip install https://github.com/ipython-contrib/jupyter_contrib_nbextensions/tarball/master
pip install jupyter_nbextensions_configurator
jupyter contrib nbextension install --user
jupyter nbextensions_configurator enable --user
pip install RISE
jupyter-nbextension install rise --py --sys-prefix
jupyter-nbextension enable rise --py --sys-prefix
pip install autopep8