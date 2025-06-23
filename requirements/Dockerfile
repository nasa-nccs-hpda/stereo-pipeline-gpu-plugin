FROM nvcr.io/nvidia/pytorch:25.05-py3

# Arguments to pass to the image
ARG STEREO_URL="https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/3.5.0/StereoPipeline-3.5.0-2025-04-28-x86_64-Linux.tar.bz2"

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p "/opt/DgStereo" && \
    wget "${STEREO_URL}" -O /opt/StereoPipeline.tar.bz2 && \
    cd /opt && \
    tar -jxf StereoPipeline.tar.bz2 && \
    rm /opt/StereoPipeline.tar.bz2 && \
    mv /opt/StereoPipeline* /opt/StereoPipeline

HEALTHCHECK NONE
ENTRYPOINT []
CMD ["/bin/bash"]
