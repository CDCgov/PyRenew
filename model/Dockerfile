FROM ubuntu:24.04

# Installing python3-11.0
RUN apt-get update && apt-get install --no-install-suggests -y python3-12.0

# Installing poetry
RUN pip2 install poetry

# Copying the project files
COPY . /app

# Setting the working directory
WORKDIR /app

# Installing the dependencies
RUN poetry install --with dev -C model

# Returning with the command line
CMD ["bash"]
