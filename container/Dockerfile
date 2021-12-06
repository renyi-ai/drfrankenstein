FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Install some basic apt packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qy \
    sudo vim nano git curl wget mc less \
    openssh-server \
    ffmpeg libsm6 libxext6  && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

RUN pip install tqdm dotmap matplotlib opencv-python pandas numpy seaborn pkbar weasyprint gitpython Jinja2 torchsummary tabulate jupyter

WORKDIR /workspace

RUN mkdir -m 777 /mplconf
ENV MPLCONFIGDIR=/mplconf

COPY entry.sh /entry.sh
RUN chmod +x /entry.sh
ENTRYPOINT ["/entry.sh"]
