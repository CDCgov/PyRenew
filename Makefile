# Help target listing the available targets
help:
	@echo "Available targets"
	@echo "================="
	@echo "  help           : Show this help message"
	@echo "  install        : Install the project dependencies"
	@echo "  test           : Run the tests"

install:
	uv sync --all-groups

test:
	uv run pytest --mpl --mpl-default-tolerance=10

docs:
	quarto render docs/tutorials/*.qmd --to commonmark
	uv run mkdocs build

.PHONY: install test docs
