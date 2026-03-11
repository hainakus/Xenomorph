'use strict';
require('dotenv').config();

const { REST, Routes, SlashCommandBuilder } = require('discord.js');
console.log('CLIENT_ID:', process.env.CLIENT_ID);
console.log('DISCORD_TOKEN exists:', !!process.env.DISCORD_TOKEN);
console.log(
  'DISCORD_TOKEN preview:',
  process.env.DISCORD_TOKEN ? process.env.DISCORD_TOKEN.slice(0, 12) : 'missing'
);
const commands = [
    new SlashCommandBuilder()
        .setName('hashrate')
        .setDescription('Current Xenom network hashrate'),

    new SlashCommandBuilder()
        .setName('daa')
        .setDescription('Current DAA score (block count on the virtual chain)'),

    new SlashCommandBuilder()
        .setName('supply')
        .setDescription('Circulating supply and % of total supply mined'),

    new SlashCommandBuilder()
        .setName('stats')
        .setDescription('Full Xenom network stats: hashrate, DAA score, supply, % mined'),
].map(cmd => cmd.toJSON());

const rest = new REST({ version: '10' }).setToken(process.env.DISCORD_TOKEN);

(async () => {
    try {
        console.log(`Registering ${commands.length} slash commands for bot ${process.env.CLIENT_ID} ...`);
        const data = await rest.put(
            Routes.applicationGuildCommands(process.env.CLIENT_ID, process.env.GUILD_ID),
            { body: commands },
        );
        console.log(`Successfully registered ${data.length} application commands.`);
    } catch (err) {
        console.error(err);
        process.exit(1);
    }
})();
