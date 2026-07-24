# Xenomorph Ecosystem - Complete Implementation

Complete implementation of Xenomorph blockchain ecosystem with UsefulPoW, API Gateway, and Smart Contracts.

## 📁 Structure

```
xenom-ecosystem/
├── smart-contracts/          # Solidity smart contracts
│   ├── InferencePayments.sol # USDT payment contract
│   └── ModelRegistry.sol     # Model registry contract
├── xenom/                   # Unified full node (Kaspa node + AI training + gRPC inference)
│   ├── src/
│   │   ├── training/        # Model management, miner WebSocket, gRPC inference
│   │   └── ...
│   └── Cargo.toml
├── seed-node/               # Standalone model/inference server (legacy, optional)
│   ├── src/
│   │   ├── consensus/       # UsefulPoW and FedAvg
│   │   ├── model/          # Model management
│   │   ├── serving/        # gRPC inference service
│   │   └── rpc/            # Borsh RPC client
│   └── Cargo.toml
├── api-gateway/             # Rust API Gateway
│   ├── src/
│   │   ├── handlers/       # HTTP handlers
│   │   ├── payments/       # USDT verification
│   │   └── state.rs        # Application state
│   └── Cargo.toml
└── proto/
    └── inference.proto     # gRPC definitions
```

## 🚀 Deployment

### Prerequisites

- Rust 1.70+
- Node.js 18+
- Solidity 0.8.20+
- Redis server
- Ethereum RPC endpoint (Polygon Mumbai for testnet)

### Smart Contracts

```bash
cd smart-contracts

# Install dependencies
npm install

# Compile contracts
npx hardhat compile

# Deploy to testnet
npx hardhat run scripts/deploy.js --network mumbai
```

### Xenom Node (Unified full node + AI training + gRPC inference)

```bash
cd xenom

# Build
cargo build --release

# Run devnet with miner WebSocket and gRPC inference
./target/release/xenom --devnet --utxoindex \
  --miner-ws-listen=0.0.0.0:17110 \
  --inference-grpc-listen=0.0.0.0:50051 \
  --models-dir=/data/models

# Run tests
cargo test
```

### Seed Node (legacy standalone model/inference server)

```bash
cd seed-node

# Build
cargo build --release

# Run standalone
./target/release/seed-node

# Run tests
cargo test
```

### API Gateway

```bash
cd api-gateway

# Build
cargo build --release

# Set environment variables
export USDT_CONTRACT_ADDRESS="0x..."
export RPC_URL="https://polygon-mumbai.infura.io/v3/YOUR_KEY"
export REDIS_URL="redis://127.0.0.1:6379"

# Run
./target/release/api-gateway

# Run tests
cargo test
```

## 🔧 Configuration

### Environment Variables

**API Gateway:**
- `USDT_CONTRACT_ADDRESS`: Deployed InferencePayments contract address
- `RPC_URL`: Ethereum RPC endpoint
- `REDIS_URL`: Redis connection string

**Seed Node:**
- `XENOMORPH_RPC`: Xenomorph blockchain RPC address (default: 127.0.0.1:16110)
- `MODEL_PATH`: Path to model storage (default: /data/models)

## 📡 API Endpoints

### API Gateway (Port 3000)

- `GET /models` - List available models
- `GET /models/:id` - Get model information
- `POST /predict/:model_id` - Make prediction (requires payment)
- `GET /queries/:id` - Get query status
- `POST /webhook/payment` - Payment webhook
- `GET /health` - Health check

### Seed Node (Port 50051)

gRPC service implementing `xenom.inference.Inference`:
- `Predict` - Make prediction
- `Embed` - Generate embeddings
- `GetModelInfo` - Get model information
- `ListModels` - List available models
- `HealthCheck` - Health check

## 💰 Payment Flow

1. Client initiates payment to InferencePayments contract
2. Payment verified by API Gateway via ethers-rs
3. Query forwarded to seed node
4. Seed node processes inference
5. Result returned to client
6. Payment distributed: 70% seed, 20% training, 10% treasury

## 🧪 Testing

```bash
# Smart contracts
cd smart-contracts
npx hardhat test

# Seed node
cd seed-node
cargo test

# API Gateway
cd api-gateway
cargo test
```

## 🔐 Security

- All model storage encrypted with AES-256-GCM
- gRPC communication uses TLS in production
- Payment verification on-chain before serving queries
- Rate limiting per wallet
- ReentrancyGuard on all contract functions

## 📊 Monitoring

- Seed node health check: `GET http://localhost:50051/health`
- API Gateway health check: `GET http://localhost:3000/health`
- Redis monitoring for cache hit rates
- Contract events for payment tracking

## 🚨 Troubleshooting

**Seed node won't start:**
- Check Xenomorph RPC connectivity
- Verify model directory permissions
- Check gRPC port availability

**API Gateway payment verification fails:**
- Verify USDT contract address
- Check RPC endpoint connectivity
- Ensure sufficient gas for contract calls

**Model loading fails:**
- Check model file integrity
- Verify encryption key
- Check storage permissions

## 📝 License

MIT License - See LICENSE file for details
