'use strict';

globalThis.WebSocket = require('websocket').w3cwebsocket;
const { RpcClient, Encoding } = require('../../wasm/nodejs/kaspa');
require('dotenv').config();

// Test multiple nodes — pass extra URLs as CLI args: node test-hashrate.js ws://a:17110 ws://b:17110
const EXTRA_URLS = process.argv.slice(2);
const NODE_URL  = process.env.NODE_URL  || 'ws://127.0.0.1:17110';
const NODES     = [NODE_URL, ...EXTRA_URLS];
const NETWORK   = process.env.NETWORK_ID || 'mainnet';
const BPS       = 4;

const fmt = (n, decimals = 4) => {
    const v = Number(n);
    if (v >= 1e18) return `${(v/1e18).toFixed(decimals)} EH/s`;
    if (v >= 1e15) return `${(v/1e15).toFixed(decimals)} PH/s`;
    if (v >= 1e12) return `${(v/1e12).toFixed(decimals)} TH/s`;
    if (v >= 1e9)  return `${(v/1e9).toFixed(decimals)} GH/s`;
    if (v >= 1e6)  return `${(v/1e6).toFixed(decimals)} MH/s`;
    return `${v.toFixed(decimals)} H/s`;
};

function queryNode(url) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('connect timeout')), 15000);
        const rpc = new RpcClient({ url, encoding: Encoding.Borsh, networkId: NETWORK });

        rpc.addEventListener('connect', async () => {
            clearTimeout(timer);
            try {
                const dag = await rpc.getBlockDagInfo();
                const daaScore   = Number(dag.virtualDaaScore || dag.daaScore || 0);
                const difficulty = dag.difficulty ?? 0;

                const windows = [1000, 2641];
                const estimates = await Promise.all(
                    windows.map(w =>
                        rpc.estimateNetworkHashesPerSecond({ windowSize: w, startHash: undefined })
                           .then(r => ({ w, hps: Number(r.networkHashesPerSecond ?? 0) }))
                           .catch(e => ({ w, hps: 0, err: e.message }))
                    )
                );

                console.log(`╔══ Node: ${url}`);
                console.log(`║  DAA score  : ${daaScore.toLocaleString()}`);
                console.log(`║  Difficulty : ${difficulty.toFixed(6)}`);
                console.log(`║  d×2×BPS (correct)  : ${fmt(difficulty * 2 * BPS)}`);
                console.log(`║  Explorer d×2 (4×↓) : ${fmt(difficulty * 2)}`);
                for (const e of estimates) {
                    if (e.err) {
                        console.log(`║  estimateHPS w=${String(e.w).padEnd(4)}: ERROR — ${e.err}`);
                    } else {
                        console.log(`║  estimateHPS w=${String(e.w).padEnd(4)}: ${fmt(e.hps)}`);
                    }
                }
                console.log('╚' + '═'.repeat(50));
                console.log('');

                await rpc.disconnect();
                resolve();
            } catch (e) {
                await rpc.disconnect().catch(() => {});
                reject(e);
            }
        });

        rpc.connect().catch(reject);
    });
}

(async () => {
    for (const url of NODES) {
        await queryNode(url).catch(e => console.error(`[${url}] failed: ${e.message}`));
    }
    process.exit(0);
})();
