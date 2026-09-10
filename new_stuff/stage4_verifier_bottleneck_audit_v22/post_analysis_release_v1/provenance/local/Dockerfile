FROM vllm/vllm-openai:v0.23.0-cu129-ubuntu2404

USER root
RUN /usr/bin/python3 -m pip install --no-cache-dir --upgrade --force-reinstall --no-deps \
    "transformers==4.57.6" \
    "peft==0.19.1" \
    "numpy==2.3.5" \
    "scipy==1.16.3" \
    "huggingface-hub==0.36.2" \
    "tokenizers==0.22.2" \
    "accelerate==1.13.0"

ENV PYTHONUNBUFFERED=1
ENTRYPOINT []
CMD ["/usr/bin/python3"]
