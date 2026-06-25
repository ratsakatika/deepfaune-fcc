# Convenience targets for the FCC batch tooling. British English; no em dashes.

PYTHON ?= python3

.PHONY: help install test compile

help:
	@echo "Targets:"
	@echo "  make install   Put 'dfrun' on PATH and add the Desktop shortcut."
	@echo "  make test      Run the unit test suite."
	@echo "  make compile   Byte-compile the tools (syntax check)."

install:
	bash install.sh

test:
	$(PYTHON) -m pytest tests/ -q

compile:
	$(PYTHON) -m py_compile deepfaune_batch.py dfrun.py detectTools.py
