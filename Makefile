.PHONY: setup build test stress logs clean health help

SHELL := /bin/bash

help:
	@echo "Xenomorph AI Devnet Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  setup   - One-command devnet setup"
	@echo "  build   - Build all devnet Docker images"
	@echo "  test    - Run health, governance, and payment tests"
	@echo "  stress  - Run a 30-minute stress test"
	@echo "  logs    - Collect logs from the last hour"
	@echo "  health  - Check devnet health"
	@echo "  clean   - Tear down the devnet"

setup:
	./scripts/quick-devnet.sh

build:
	./scripts/build-devnet.sh --parallel

health:
	./scripts/check-devnet-health.sh

test: health
	./scripts/test-governance-flow.sh --proposal add-model --model-id test-v1
	./scripts/test-payment-flow.sh --amount 0.1

stress:
	./scripts/stress-test.sh --miners 5 --duration 30m

logs:
	./scripts/logs-collector.sh --since 1h

clean:
	./scripts/cleanup-devnet.sh --volumes
