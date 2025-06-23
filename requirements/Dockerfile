FROM nvcr.io/nvidia/pytorch:25.05-py3

# Arguments to pass to the image
ARG STEREO_URL="https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/3.5.0/StereoPipeline-3.5.0-2025-04-28-x86_64-Linux.tar.bz2"
ENV PATH="$PATH:/opt/StereoPipeline/bin"
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get -y dist-upgrade && \
    apt-get install -y --no-install-recommends \
        wget vim curl git procps gcc g++ bzip2 libssl-dev \
        libsqlite3-dev libx11-dev libgeos++-dev libproj-dev \
        build-essential parallel libdatetime-perl gawk util-linux bc \
        libgdbm-dev libc6-dev libbz2-dev libffi-dev \
        zlib1g-dev liblzma-dev libgirepository1.0-dev libcairo2-dev \
        pkg-config gir1.2-gtk-3.0 libnetcdf-dev libhdf4-dev && \
    apt-get -y autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apt

# Install Stereo Pipeline
RUN mkdir -p "/opt/DgStereo" && \
    wget "${STEREO_URL}" -O /opt/StereoPipeline.tar.bz2 && \
    cd /opt && \
    tar -jxf StereoPipeline.tar.bz2 && \
    rm /opt/StereoPipeline.tar.bz2 && \
    mv /opt/StereoPipeline* /opt/StereoPipeline

RUN pip install rioxarray

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
