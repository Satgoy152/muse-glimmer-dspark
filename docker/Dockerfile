# vLLM nightly (Glimmer support has not shipped in a stable release) with the
# community DSpark drafter baked in.
#
# The target model (~60 GB) is NOT baked in: it is pulled at runtime into
# HF_HOME, which must be a mounted persistent volume or every container
# restart re-downloads it.

# Pin by digest, not the floating :nightly tag -- an unpinned base will drift
# under a training run and change acceptance numbers with no other signal.
ARG VLLM_BASE=vllm/vllm-openai:nightly
FROM ${VLLM_BASE}

USER root

# Nightly carries the dflash method. Whether it also carries dspark is
# unverified; install speculators only if the baked drafter needs it.
ARG SPECULATORS_REF=
RUN if [ -n "${SPECULATORS_REF}" ]; then \
      pip install --no-cache-dir \
        "git+https://github.com/vllm-project/speculators@${SPECULATORS_REF}"; \
    fi

# Bake the drafter (small) so it is present even on a cold volume.
ARG SPECULATOR_REPO=DaoCloud/Muse-Glimmer-30B-DSpark
ENV SPECULATOR_PATH=/opt/speculators/dspark
RUN pip install --no-cache-dir "huggingface_hub[cli]" && \
    hf download "${SPECULATOR_REPO}" --local-dir "${SPECULATOR_PATH}"

ENV HF_HOME=/root/.cache/huggingface
ENV VLLM_LOGGING_LEVEL=INFO

COPY docker/serve.sh /opt/serve.sh
RUN chmod +x /opt/serve.sh

ENTRYPOINT ["/opt/serve.sh"]
