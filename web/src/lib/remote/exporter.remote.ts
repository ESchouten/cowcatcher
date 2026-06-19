import { command, form, query } from '$app/server';
import { redirect } from '@sveltejs/kit';
import { getConfig, saveConfig } from './config.remote';
import * as v from 'valibot';

function getRedirectTarget(next: string | undefined, fallback: string) {
	return next?.startsWith('/') && !next.startsWith('//') ? next : fallback;
}

export const getTelegrams = query(async () => {
	const { app } = await getConfig();
	return app.telegrams;
});

export const saveTelegram = form(
	v.object({
		original: v.optional(v.string()),
		label: v.string(),
		token: v.string(),
		chat: v.string(),
		next: v.optional(v.string())
	}),
	async ({ label, token, chat, original, next }) => {
		const { config, app } = await getConfig();
		let found = false;
		app.telegrams.forEach((telegram) => {
			if (telegram.label === original) {
				telegram.label = label;
				telegram.token = token;
				telegram.chat = chat;
				found = true;
			}
		});
		if (!found) {
			app.telegrams.push({ label, token, chat });
		}
		await saveConfig({ config, app });
		redirect(302, getRedirectTarget(next, '/notifications'));
	}
);

export const deleteTelegram = command(
	v.object({
		label: v.string()
	}),
	async ({ label }) => {
		const { config, app } = await getConfig();
		const telegram = app.telegrams.find((telegram) => telegram.label === label);
		app.telegrams = app.telegrams.filter((telegram) => telegram.label !== label);
		if (telegram) {
			config.detectors.forEach((detector) => {
				if (detector.exporters?.telegram) {
					detector.exporters.telegram = detector.exporters.telegram.filter(
						(t) => t.token !== telegram.token || t.chat !== telegram.chat
					);
				}
			});
		}
		await saveConfig({ config, app });
	}
);

export const testTelegram = command(
	v.object({
		token: v.string(),
		chat: v.string()
	}),
	async ({ token, chat }) => {
		const response = await fetch('https://api.telegram.org/bot' + token + '/sendMessage', {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json'
			},
			body: JSON.stringify({
				chat_id: chat,
				text: 'Test notification'
			})
		});
		return response.json();
	}
);
