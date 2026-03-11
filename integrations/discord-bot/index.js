'use strict';

// WebSocket shim required by the WASM RPC client in Node.js
globalThis.WebSocket = require('websocket').w3cwebsocket;

// Require the Xenomorph WASM build directly (wasm-pack output: wasm/nodejs/kaspa)
const { RpcClient, Encoding } = require('../../wasm/nodejs/kaspa');

require('dotenv').config();

const { Client, GatewayIntentBits, EmbedBuilder, ActivityType } = require('discord.js');

// ── Config ────────────────────────────────────────────────────────────────────
const NODE_URL    = process.env.NODE_URL    || 'ws://127.0.0.1:17110';
const NETWORK_ID  = process.env.NETWORK_ID  || 'mainnet';
const CACHE_TTL   = parseInt(process.env.CACHE_TTL_MS || '15000', 10); // 15 s

// Max supply: 28.7 billion XENOM (same emission schedule as Kaspa)
const SOMPI_PER_XENOM = 100_000_000n;

// ── RPC client ────────────────────────────────────────────────────────────────
const rpc = new RpcClient({
    url:      NODE_URL,
    encoding: Encoding.Borsh,
    networkId: NETWORK_ID,
});

let rpcReady = false;

rpc.addEventListener('connect', () => {
    rpcReady = true;
    console.log(`[RPC] Connected to ${NODE_URL}`);
});

rpc.addEventListener('disconnect', () => {
    rpcReady = false;
    console.warn('[RPC] Disconnected — reconnecting...');
});

(async () => {
    try {
        await rpc.connect();
    } catch (e) {
        console.error('[RPC] Initial connect failed:', e.message);
    }
})();

// ── Stats cache ───────────────────────────────────────────────────────────────
let cache = null;
let cacheTs = 0;

async function fetchStats() {
    const now = Date.now();
    if (cache && now - cacheTs < CACHE_TTL) return cache;

    if (!rpcReady) throw new Error('Node not connected. Check `NODE_URL` in .env');

    const [dagInfo, supplyRes, hpsRes] = await Promise.all([
        rpc.getBlockDagInfo(),
        rpc.getCoinSupply(),
        // MIN_WINDOW_SIZE=1000 enforced in difficulty.rs — 1000 blocks = 250s at 4 BPS
        rpc.estimateNetworkHashesPerSecond({ windowSize: 1000, startHash: undefined }),
    ]);

    const daaScore         = BigInt(dagInfo.virtualDaaScore ?? dagInfo.daaScore ?? 0n);
    const difficulty       = dagInfo.difficulty ?? 0;
    const maxSompi         = BigInt(supplyRes.maxSompi);
    const circulatingSompi = BigInt(supplyRes.circulatingSompi);
    const rawHps           = hpsRes.networkHashesPerSecond ?? 0;
    const hashesPerSecond  = typeof rawHps === 'bigint' ? rawHps : BigInt(Math.round(Number(rawHps)));

    cache = { daaScore, difficulty, maxSompi, circulatingSompi, hashesPerSecond };
    cacheTs = now;
    return cache;
}

// ── Formatters ────────────────────────────────────────────────────────────────
function fmtHashrate(hps) {
    const n = Number(hps);
    if (n >= 1e18) return `${(n / 1e18).toFixed(2)} EH/s`;
    if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PH/s`;
    if (n >= 1e12) return `${(n / 1e12).toFixed(2)} TH/s`;
    if (n >= 1e9)  return `${(n / 1e9).toFixed(2)} GH/s`;
    if (n >= 1e6)  return `${(n / 1e6).toFixed(2)} MH/s`;
    if (n >= 1e3)  return `${(n / 1e3).toFixed(2)} KH/s`;
    return `${n.toFixed(0)} H/s`;
}

function fmtXenom(sompi) {
    const xenom = Number(sompi) / Number(SOMPI_PER_XENOM);
    return xenom.toLocaleString('en-US', { maximumFractionDigits: 2 }) + ' XENOM';
}

function fmtPct(circulating, max) {
    if (max === 0n) return '0.00%';
    return ((Number(circulating) / Number(max)) * 100).toFixed(4) + '%';
}

function fmtDaaScore(n) {
    return n.toLocaleString('en-US');
}

// ── Embed builders ────────────────────────────────────────────────────────────
const XENOM_COLOR = 0x00e5ff;   // cyan accent

function hashrateEmbed(stats) {
    return new EmbedBuilder()
        .setColor(XENOM_COLOR)
        .setTitle('⛏️  Xenom Network Hashrate')
        .addFields({ name: 'Hashrate', value: `**${fmtHashrate(stats.hashesPerSecond)}**`, inline: false })
        .setFooter({ text: 'Estimated over last 1000 blocks • Xenom Network' })
        .setTimestamp();
}

function daaEmbed(stats) {
    return new EmbedBuilder()
        .setColor(XENOM_COLOR)
        .setTitle('📊  DAA Score')
        .addFields(
            { name: 'DAA Score',   value: `**${fmtDaaScore(stats.daaScore)}**`, inline: true },
            { name: 'Difficulty',  value: `**${Number(stats.difficulty).toExponential(4)}**`, inline: true },
        )
        .setFooter({ text: 'Virtual chain score • Xenom Network' })
        .setTimestamp();
}

function supplyEmbed(stats) {
    return new EmbedBuilder()
        .setColor(XENOM_COLOR)
        .setTitle('💰  Circulating Supply')
        .addFields(
            { name: 'Circulating',   value: `**${fmtXenom(stats.circulatingSompi)}**`,               inline: true },
            { name: 'Max Supply',    value: `**${fmtXenom(stats.maxSompi)}**`,                        inline: true },
            { name: '% Mined',       value: `**${fmtPct(stats.circulatingSompi, stats.maxSompi)}**`,  inline: true },
        )
        .setFooter({ text: 'Requires node with --utxoindex • Xenom Network' })
        .setTimestamp();
}

function statsEmbed(stats) {
    return new EmbedBuilder()
        .setColor(XENOM_COLOR)
        .setTitle('🌐  Xenom Network Stats')
        .addFields(
            { name: '⛏️  Hashrate',       value: `**${fmtHashrate(stats.hashesPerSecond)}**`,                inline: false },
            { name: '📊  DAA Score',      value: `**${fmtDaaScore(stats.daaScore)}**`,                       inline: true  },
            { name: '📈  Difficulty',     value: `**${Number(stats.difficulty).toExponential(4)}**`,          inline: true  },
            { name: '💰  Circulating',    value: `**${fmtXenom(stats.circulatingSompi)}**`,                   inline: true  },
            { name: '🏦  Max Supply',     value: `**${fmtXenom(stats.maxSompi)}**`,                          inline: true  },
            { name: '⚡  % Mined',        value: `**${fmtPct(stats.circulatingSompi, stats.maxSompi)}**`,    inline: true  },
        )
        .setFooter({ text: 'Cached 15 s • Powered by Xenomorph wRPC WASM' })
        .setTimestamp();
}

// ── Discord client ────────────────────────────────────────────────────────────
const client = new Client({ intents: [GatewayIntentBits.Guilds] });

client.once('clientReady', () => {
    console.log(`[Discord] Logged in as ${client.user.tag}`);
    client.user.setActivity('xenom network', { type: ActivityType.Watching });
});

client.on('interactionCreate', async (interaction) => {
    if (!interaction.isChatInputCommand()) return;

    await interaction.deferReply();

    try {
        const stats = await fetchStats();

        switch (interaction.commandName) {
            case 'hashrate':
                await interaction.editReply({ embeds: [hashrateEmbed(stats)] });
                break;
            case 'daa':
                await interaction.editReply({ embeds: [daaEmbed(stats)] });
                break;
            case 'supply':
                await interaction.editReply({ embeds: [supplyEmbed(stats)] });
                break;
            case 'stats':
                await interaction.editReply({ embeds: [statsEmbed(stats)] });
                break;
            default:
                await interaction.editReply('Unknown command.');
        }
    } catch (err) {
        console.error(`[cmd:${interaction.commandName}]`, err);
        await interaction.editReply({
            embeds: [
                new EmbedBuilder()
                    .setColor(0xff4444)
                    .setTitle('❌  Error')
                    .setDescription(`\`\`\`${err.message}\`\`\``)
                    .setFooter({ text: 'Check NODE_URL in .env and that the node is running' }),
            ],
        });
    }
});

client.login(process.env.DISCORD_TOKEN).catch(err => {
    console.error('[Discord] Login failed:', err.message);
    process.exit(1);
});
