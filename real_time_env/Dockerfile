FROM nvcr.io/nvidia/pytorch:21.10-py3

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# for dlib
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# for dlib
RUN apt-get install -y build-essential cmake
# for opencv
RUN apt-get install -y libopencv-dev

WORKDIR /workspace

# install requirements
COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt
