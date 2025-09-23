FROM mambaorg/micromamba:1.5.10

# Copy your environment file into the image
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml

# Create the env named "app" from the file
RUN micromamba create -y -n app -f /tmp/environment.yml && \
    micromamba clean -a -y

# Make the env active for subsequent commands and at runtime
SHELL ["/bin/bash", "-lc"]
ENV PATH=/opt/conda/envs/app/bin:$PATH

WORKDIR /workspace
CMD ["python", "--version"]
