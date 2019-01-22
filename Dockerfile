FROM jupyter/datascience-notebook

USER root

# GoogleAnalyticsClient のインストール
RUN pip install --upgrade google-api-python-client

# vimキーバインドのプラグインインストール
RUN jupyter labextension install jupyterlab_vim

# 黒背景設定を追加
RUN mkdir -p /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
RUN echo '{"theme":"JupyterLab Dark"}' > \
  /home/jovyan/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings

USER jovyan
WORKDIR /home/jovyan
