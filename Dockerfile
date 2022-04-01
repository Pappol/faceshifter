FROM nvcr.io/nvidia/pytorch:21.10-py3

# install requirements
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt
RUN torch==1.11.0+cu113 torchvision==0.12.0+cu113