FROM mambaorg/micromamba:1.5.10
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba create -y -n app -f /tmp/environment.yml && \
    micromamba clean -a -y

SHELL ["/bin/bash", "-lc"]
ENV PATH=/opt/conda/envs/app/bin:$PATH

WORKDIR /workspace
CMD ["python", "--version"]
