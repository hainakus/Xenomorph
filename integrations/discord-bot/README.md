# Xenom Discord Bot

Discord bot for live Xenom network statistics using the **Xenomorph wRPC WASM** node bindings.

## Commands

| Command | Description |
|---------|-------------|
| `/stats` | All stats in one embed: hashrate, DAA score, difficulty, supply, % mined |
| `/hashrate` | Current network hashrate (H/s → KH/s → MH/s → GH/s → TH/s auto-scaled) |
| `/daa` | Current DAA score and difficulty |
| `/supply` | Circulating supply, max supply and % mined |

Stats are cached for 15 seconds to avoid spamming the node.

---

## Prerequisites

- Node.js ≥ 18
- A running Xenom node with **wRPC enabled** and optionally `--utxoindex` (required for `/supply`)
- A Discord application/bot at <https://discord.com/developers/applications> (bot ID: `1294650332491546665`)
- The Xenomorph WASM node build (see step 1 below)

### Enable wRPC on the Xenom node

Add to your node launch command:
```bash
xenom --rpclisten-borsh 0.0.0.0:17110 --utxoindex
```

---

## Setup

### 1. Build the Xenomorph WASM (once)

The bot loads the WASM directly from `wasm/nodejs/kaspa/` (wasm-pack build output). You must build it first:

```bash
# From the repo root — install wasm-pack if needed:
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the Node.js WASM target
cd wasm
bash build-node        # output → wasm/nodejs/kaspa/
```

After the build, `wasm/nodejs/kaspa/kaspa.js` and `kaspa_bg.wasm` will be present.

### 2. Install bot dependencies

```bash
cd integrations/discord-bot
npm install
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your values
```

| Variable | Default | Description |
|----------|---------|-------------|
| `DISCORD_TOKEN` | *(required)* | Bot token from Discord developer portal |
| `CLIENT_ID` | `1294650332491546665` | Your bot application ID |
| `NODE_URL` | `ws://127.0.0.1:17110` | Xenom node wRPC endpoint |
| `NETWORK_ID` | `mainnet` | `mainnet`, `testnet-1`, or `devnet` |
| `CACHE_TTL_MS` | `15000` | Stats cache lifetime in milliseconds |

### 3. Register slash commands

```bash
npm run deploy
```

This registers `/stats`, `/hashrate`, `/daa`, `/supply` globally. Takes up to 1 hour to propagate to all servers (instant in the bot's own server if using guild commands).

### 4. Start the bot

```bash
npm start
```

---

## Discord Developer Portal — Invite URL

In the **Discord Developer Portal → your app → OAuth2 → URL Generator**:

- **Scopes:** `bot`, `applications.commands`
- **Bot permissions:** `Send Messages`, `Use Slash Commands`, `Embed Links`

Or build the URL manually:
```
https://discord.com/api/oauth2/authorize?client_id=1294650332491546665&permissions=277025392640&scope=bot%20applications.commands
```

---

## Running as a service (systemd)

```ini
# /etc/systemd/system/xenom-discord-bot.service
[Unit]
Description=Xenom Discord Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/xenom-discord-bot
ExecStart=/usr/bin/node index.js
Restart=on-failure
RestartSec=10
EnvironmentFile=/opt/xenom-discord-bot/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now xenom-discord-bot
sudo journalctl -fu xenom-discord-bot
```

---

## How it works

The bot uses **`kaspa-wasm`** (the Xenomorph WASM SDK compiled to Node.js) to open a persistent wRPC WebSocket connection to the Xenom node. On each command it calls three RPC methods in parallel:

| RPC Method | Returns |
|------------|---------|
| `estimateNetworkHashesPerSecond({ windowSize: 1000 })` | `networkHashesPerSecond` |
| `getBlockDagInfo()` | `virtualDaaScore`, `difficulty` |
| `getCoinSupply()` | `maxSompi`, `circulatingSompi` |

Results are cached for `CACHE_TTL_MS` milliseconds so multiple users hitting commands simultaneously only trigger one node query.
