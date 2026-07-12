export const STAGES = ['approved', 'rejected', 'unvalidated'] as const;
export type Stage = (typeof STAGES)[number];

export const DEFAULT_SCHEMA_URL =
	'https://raw.githubusercontent.com/ESchouten/ai-detector/main/config/config.schema.json';

export interface IdentityResult {
	identity: string;
	similarity: number;
}

export interface Metadata {
	type: string;
	timestamp: string;
	validated: boolean | null;
	confidence: number;
	confidences: Record<string, number>;
	identities: IdentityResult[];
	detections: number;
	start: string;
	end: string;
	duration: number;
	crop?: { x1: number; y1: number; x2: number; y2: number } | null;
	crops?: Array<Record<string, unknown>>;
}

export interface CowIdentityConfig {
	database: string;
	enrollment?: {
		identity_count?: number;
	};
	segment_model?: string;
	imgsz?: number;
	confidence?: number;
	match_threshold?: number;
	match_margin?: number;
	min_area_ratio?: number;
	max_area_ratio?: number;
	margin?: number;
	nms_iou?: number;
	track_samples?: number;
	track_max_age?: number;
}

export interface DetectorConfig {
	detection: {
		source: string[];
		[key: string]: unknown;
	};
	yolo?: {
		model: string;
		confidence: number | Record<string, number>;
		task?: 'detect' | 'segment';
		tracking?: boolean;
		tracker?: string;
		imgsz?: number;
		frames_min?: number;
	};
	identity?: CowIdentityConfig;
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
