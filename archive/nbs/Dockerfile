FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
        wget \
        git \
        zsh \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /my-app-ap/

COPY notes /my-app-ap/

CMD ["/bin/bash"]
# CMD ["python", "model/classify_iris.py"]
