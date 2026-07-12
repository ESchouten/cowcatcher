import type {
	ChatConfig,
	Config as ConfigDocument,
	DetectionConfig as DetectionDocument,
	DetectorConfig as DetectorDocument,
	DiskConfig,
	ExportersConfig as ExportersDocument,
	SSEConfig,
	WebhookConfig
} from '$lib/generated/config';
import type { DetectionMetadata as MetadataDocument } from '$lib/generated/metadata';

export const STAGES = ['approved', 'rejected', 'unvalidated'] as const;
export type Stage = (typeof STAGES)[number];

export type Metadata = MetadataDocument;
export type DetectionMetadata = Metadata & {
	type: string;
	stage: Stage;
	locator: string;
};

export const DEFAULT_SCHEMA_URL =
	'https://raw.githubusercontent.com/ESchouten/ai-detector/main/config/config.schema.json';

export type TelegramConfig = ChatConfig;
export type { SSEConfig };

type NormalizedExporters = Omit<ExportersDocument, 'disk' | 'telegram' | 'webhook' | 'sse'> & {
	disk?: DiskConfig[];
	telegram?: TelegramConfig[];
	webhook?: WebhookConfig[];
	sse?: SSEConfig[];
};

export interface DetectorConfig extends Omit<DetectorDocument, 'detection' | 'exporters'> {
	detection: Omit<DetectionDocument, 'source'> & { source: string[] };
	exporters?: NormalizedExporters;
}

export interface Config extends Omit<ConfigDocument, 'detectors'> {
	$schema?: string;
	detectors: DetectorConfig[];
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
