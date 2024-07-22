CONTAINER=podman
# Pass as environment variables
# CONTAINER_REGISTRY=
# CONTAINER_URL=

# Help target listing the available targets
help:
	@echo "Available targets"
	@echo "================="
	@echo "  help           : Show this help message"
	@echo "  install        : Install the project dependencies"
	@echo "  test           : Run the tests"
	@echo "  test_images    : Generate reference images for tests"
	@echo "  image-build    : Build the podman/Docker image"
	@echo "  image-run      : Run the podman/Docker image"
	@echo "  image-push     : Push the podman/Docker image to the registry"

install:
	poetry install

test:
	poetry run pytest --mpl --mpl-default-tolerance=10

image-build: Dockerfile
	$(CONTAINER) build -t pyrenew:latest .

image-run:
	$(CONTAINER) run -it --rm -v $(PWD):/mnt pyrenew:latest

image-az-login:
	if [ -z $(CONTAINER_REGISTRY) ]; then \
		echo "Please set the CONTAINER_REGISTRY environment variable"; \
		exit 1; \
	fi
	az login
	az acr login --name $(CONTAINER_REGISTRY)

image-push:
	if [ -z $(CONTAINER_URL) ]; then \
		echo "Please set the CONTAINER_URL environment variable"; \
		exit 1; \
	fi
	$(CONTAINER) push pyrenew:latest $(CONTAINER_URL)/pyrenew:latest

.PHONY: install test test_images image-build image-run
