export const STAGES = ['approved', 'rejected', 'unvalidated'] as const;
export type Stage = (typeof STAGES)[number];

export const DEFAULT_SCHEMA_URL =
	'https://raw.githubusercontent.com/ESchouten/ai-detector/main/config/config.schema.json';

export interface DetectorConfig {
	detection: {
		source: string[];
		[key: string]: unknown;
	};
	yolo?: {
		model: string;
		confidence: number;
		tracking?: boolean;
		frames_min: number;
	};
	exporters?: {
		telegram?: TelegramConfig[];
		sse?: SSEConfig[];
		[key: string]: unknown[] | undefined;
	};
	[key: string]: unknown;
}

export interface TelegramConfig {
	token: string;
	chat: string;
	alert_every?: number;
}

export interface SSEConfig {
	port?: number;
	endpoint?: string | null;
}

export interface Config {
	$schema?: string;
	detectors: DetectorConfig[];
	[key: string]: unknown;
}

export interface AppConfig {
	streams: StreamMeta[];
	telegrams: TelegramMeta[];
	detectors: DetectorMeta[];
}

export interface DetectorMeta {
	label: string;
}

export interface TelegramMeta extends TelegramConfig {
	label: string;
}

export interface StreamMeta {
	label?: string;
	source: string;
}
