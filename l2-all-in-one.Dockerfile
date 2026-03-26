FROM rust:1.85-bookworm AS builder

WORKDIR /src
COPY . .

RUN apt-get update && apt-get install -y --no-install-recommends \
  clang \
  libclang-dev \
  protobuf-compiler \
  && rm -rf /var/lib/apt/lists/*

RUN cargo build --release \
  -p genetics-l2-coordinator \
  -p genetics-l2-fetcher \
  -p genetics-l2-worker \
  -p genetics-l2-validator \
  -p genetics-l2-settlement \
  -p xenom-anchor-committer \
  -p xenom-evm-node

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  bash \
  ca-certificates \
  curl \
  tini \
  python3 \
  python3-pip \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
  numpy \
  pandas \
  requests

RUN mkdir -p /opt/xenom/bin /opt/xenom/scripts /var/lib/xenom /var/log/xenom

COPY --from=builder /src/target/release/genetics-l2-coordinator /opt/xenom/bin/
COPY --from=builder /src/target/release/genetics-l2-fetcher /opt/xenom/bin/
COPY --from=builder /src/target/release/genetics-l2-worker /opt/xenom/bin/
COPY --from=builder /src/target/release/genetics-l2-validator /opt/xenom/bin/
COPY --from=builder /src/target/release/genetics-l2-settlement /opt/xenom/bin/
COPY --from=builder /src/target/release/xenom-anchor-committer /opt/xenom/bin/
COPY --from=builder /src/target/release/xenom-evm-node /opt/xenom/bin/
COPY --from=builder /src/scripts /opt/xenom/scripts
COPY l2-all-in-one-entrypoint.sh /opt/xenom/l2-entrypoint.sh

RUN chmod +x /opt/xenom/l2-entrypoint.sh /opt/xenom/bin/*

ENV DB_PATH=/var/lib/xenom/genetics-l2.db
ENV WORK_ROOT=/var/lib/xenom/l2-work
ENV COORDINATOR_PORT=8091
ENV COORDINATOR_URL=http://127.0.0.1:8091
ENV FETCHER_POLL_SECS=300
ENV WORKER_POLL_MS=5000
ENV VALIDATOR_POLL_MS=10000
ENV VALIDATOR_SCORE_TOLERANCE=0.05
ENV SETTLEMENT_POLL_MS=15000
ENV FETCHER_FLAGS="--sra --igsr --gnomad --gdc --clinvar"
ENV ENABLE_WORKER=0
ENV ENABLE_EVM=1
ENV EVM_RPC_ADDR=127.0.0.1:8545
ENV EVM_BLOCK_TIME_MS=2000
ENV EVM_STATE_DIR=/var/lib/xenom/evm-state
ENV EVM_DEVNET=0
ENV EVM_L1_NODE=
ENV SETTLEMENT_MODE=dry-run
ENV SETTLEMENT_NODE=127.0.0.1:36669
ENV SETTLEMENT_NETWORK=mainnet
ENV SETTLEMENT_EVM_NODE=http://127.0.0.1:8545
ENV SETTLEMENT_FEE_SOMPI=
ENV SETTLEMENT_QUORUM=1
ENV SETTLEMENT_SCORE_TOLERANCE=0.05
ENV ENABLE_ANCHOR_COMMITTER=1
ENV ANCHOR_MODE=dry-run
ENV ANCHOR_NODE=127.0.0.1:36669
ENV ANCHOR_EVM_NODE=http://127.0.0.1:8545
ENV ANCHOR_POLL_MS=10000
ENV ANCHOR_STATE_DIR=/var/lib/xenom/anchor-state
ENV ANCHOR_NETWORK=mainnet
ENV AIO_IMAGE_TAG=iotapi322/xenom-l2-aio:v2

VOLUME ["/var/lib/xenom", "/var/log/xenom"]

EXPOSE 8091 8545

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/xenom/l2-entrypoint.sh"]
