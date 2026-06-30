export const STAGES = ['approved', 'rejected', 'unvalidated'] as const;
export type Stage = (typeof STAGES)[number];

export const DEFAULT_SCHEMA_URL =
	'https://raw.githubusercontent.com/ESchouten/ai-detector/main/config/config.schema.json';

export interface IdentityResult {
	provider: string;
	identity_id: string | null;
	name: string | null;
	status: 'matched' | 'created' | 'unknown';
	similarity?: number | null;
}

export interface Crop {
	x1: number;
	y1: number;
	x2: number;
	y2: number;
}

export interface Metadata {
	type: string;
	timestamp: string;
	validated: boolean | null;
	confidence: number;
	confidences: Record<string, number>;
	identity: IdentityResult | null;
	identities: IdentityResult[];
	detections: number;
	start: string;
	end: string;
	duration: number;
	crop?: Crop | null;
	crops?: Crop[];
}

export interface DetectorConfig {
	detection: {
		source: string[];
		[key: string]: unknown;
	};
	yolo?: {
		model: string;
		confidence: number;
		frames_min: number;
	};
	exporters?: {
		telegram?: TelegramConfig[];
		[key: string]: unknown[] | undefined;
	};
	identity?: DetectorIdentityConfig | null;
	[key: string]: unknown;
}

export interface IdentityProviderConfig {
	id: string;
	type?: 'wildlife_tools';
	database: string;
	model?: string;
	segment_model?: string | null;
	segment_labels?: string[];
	segment_confidence?: number;
	segment_imgsz?: number;
	[key: string]: unknown;
}

export interface IdentityConfig {
	providers: IdentityProviderConfig[];
}

export interface DetectorIdentityConfig {
	provider: string;
	multiple?: boolean;
}

export interface TelegramConfig {
	token: string;
	chat: string;
	alert_every?: number;
}

export interface Config {
	$schema?: string;
	detectors: DetectorConfig[];
	identity?: IdentityConfig | null;
	[key: string]: unknown;
}

export interface AppConfig {
	streams: StreamMeta[];
	telegrams: TelegramMeta[];
	detectors: DetectorMeta[];
	identities: IdentityMeta[];
}

export interface DetectorMeta {
	label: string;
}

export interface TelegramMeta extends TelegramConfig {
	label: string;
}

export interface IdentityMeta extends IdentityProviderConfig {
	label: string;
}

export interface StreamMeta {
	label?: string;
	source: string;
}
