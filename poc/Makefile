.PHONY: install test lint check smoke all

install:
	pip install -e .[dev]

test:
	pytest -q

lint:
	ruff check . --fix && ruff format .

check:
	ruff check . && ruff format --check .

smoke:
	cmlis bench --simulate --workloads short --configs naive,full --reps 3

all: install lint test
