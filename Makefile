.PHONY: test test-unit test-integration

test: test-unit test-integration

test-unit:
	python3 -m pytest tests/unit/ -v --tb=short

test-integration:
	python3 -m pytest tests/integration/ -v --tb=short
