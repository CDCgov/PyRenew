# Help target listing the available targets
help:
	@echo "Available targets"
	@echo "================="
	@echo "  help           : Show this help message"
	@echo "  install        : Install the project dependencies"
	@echo "  test           : Run the tests"

install:
	poetry install

test:
	poetry run pytest --mpl --mpl-default-tolerance=10


.PHONY: install test
