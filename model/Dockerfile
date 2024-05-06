FROM ubuntu:24.04

# Installing python3-11.0
RUN apt-get update && apt install -y python3.12

# Adding the alias python3 for python3.12
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Installing poetry
RUN apt install -y pipx && \
    pipx install poetry

RUN echo "export PATH=$PATH:/root/.local/bin" >> ~/.bashrc

ENV PATH="${PATH}:/root/.local/bin"

# Copying the project files
COPY . /app

# Setting the working directory
WORKDIR /app

# Installing the dependencies
RUN poetry install

# Returning with the command line
CMD ["bash"]
