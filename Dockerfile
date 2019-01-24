FROM jupyter/datascience-notebook

USER root

# GoogleAnalyticsClient のインストール
RUN pip install --upgrade google-api-python-client redash-dynamic-query ipython-sql

# vimキーバインドのプラグインインストール
RUN jupyter labextension install jupyterlab_vim

# 黒背景設定を追加
RUN mkdir -p /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
RUN echo '{"theme":"JupyterLab Dark"}' > \
  /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

RUN mkdir /home/jovyan/notebooks
RUN chown jovyan:users /home/jovyan/notebooks

USER jovyan
WORKDIR /home/jovyan
