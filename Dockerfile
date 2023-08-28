FROM mister5ive/ai.deploy.box:1.1.0
MAINTAINER "Hulk Wang"

LABEL description="AIDeployBox:A toolbox for deep learning model deployment using C++."

ARG AiDB_HOME=/aidb
ARG LINUX_LIBS=linux.zip
ARG MODEL=models-lite.zip

ARG LIBS_URL=https://github.com/TalkUHulk/ai.deploy.box/releases/download/1.1.0/$LINUX_LIBS
ARG MODEL_LITE_URL=https://github.com/TalkUHulk/ai.deploy.box/releases/download/1.1.0/$MODEL

ADD . $AiDB_HOME/

RUN curl -LJO $LIBS_URL \
    && curl -LJO $MODEL_LITE_URL \
    && unzip $LINUX_LIBS -d $AiDB_HOME/libs/ \
    && rm -f $LINUX_LIBS \
    && unzip $MODEL -d $AiDB_HOME/ \
    && rm -f $MODEL \
    && mv $AiDB_HOME/models-lite $AiDB_HOME/models \
    && mkdir $AiDB_HOME/build \
    && cd $AiDB_HOME/build \
    && cmake -DENGINE_MNN=ON -DENGINE_ORT=ON -DENGINE_NCNN=ON -DENGINE_TNN=ON -DENGINE_OPV=ON -DENGINE_PPLite=ON -DBUILD_SAMPLE=ON ../ \
    && make --jobs=$(nproc --all)

ENV LD_LIBRARY_PATH $AiDB_HOME/libs/linux/x86_64:$AiDB_HOME/libs/linux/x86_64/tnn:$AiDB_HOME/libs:$AiDB_HOME/libs/linux/x86_64/openvino:$AiDB_HOME/libs/linux/x86_64/paddlelite:$LD_LIBRARY_PATH

WORKDIR $AiDB_HOME/

