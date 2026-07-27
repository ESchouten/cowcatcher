/* Generated from config JSON Schema. Do not edit manually. */

/**
 * @minItems 1
 */
export type Detectors = [DetectorConfig, ...DetectorConfig[]];
export type Source = string | [string, ...string[]];
export type Interval = number;
export type FrameRetention = number;
export type FramesWidth = number;
export type Model = string;
export type Task = 'detect' | 'segment';
export type Confidence =
	| number
	| {
			[k: string]: number;
	  };
export type Tracking = boolean;
export type TimeMax = number;
export type Timeout = number;
export type Cooldown =
	| number
	| {
			[k: string]: number;
	  };
export type IncludeTrailingTime = number;
export type FramesMin = number;
export type Imgsz = number;
export type Iou = number;
export type Tracker = string;
export type Database = string;
export type TargetLabel = string;
export type Singular = string;
export type Plural = string;
export type OfficialIdLabel = string;
export type MinAreaRatio = number;
export type MaxAreaRatio = number;
export type FrameEdgeMargin = number;
export type ZoneId = string;
export type X1 = number;
export type Y1 = number;
export type X2 = number;
export type Y2 = number;
export type MinimumBoxInsideRatio = number;
export type MinimumStableFrames = number;
export type ClearFrames = number;
export type Encoder = 'miewid-dual-crop-v1';
export type SimilarityThreshold = number;
export type SimilarityMargin = number;
export type QueryFrames = number;
export type GalleryFrames = number;
export type TrackMaxAge = number;
export type Vlm = VLMConfig | VLMConfig[] | null;
export type Prompt = string;
export type Model1 = string | [string, ...string[]];
export type Key = string | null;
export type Url = string | null;
export type Strategy = 'IMAGE' | 'VIDEO';
export type CropPadding = number;
export type Timeout1 = number;
export type Disk = DiskConfig | DiskConfig[] | null;
export type Confidence1 =
	| number
	| {
			[k: string]: number;
	  }
	| null;
export type CropPadding1 = number;
export type ExportRejected = boolean;
export type Directory = string | null;
export type Strategy1 = 'ALL' | 'BEST';
export type Telegram = ChatConfig | ChatConfig[] | null;
export type Confidence2 =
	| number
	| {
			[k: string]: number;
	  }
	| null;
export type CropPadding2 = number;
export type ExportRejected1 = boolean;
export type IncludeImage = boolean;
export type IncludePlot = boolean;
export type IncludeCrop = boolean;
export type IncludeVideo = boolean;
export type VideoWidth = number | null;
export type VideoCrf = number;
export type Token = string;
export type Chat = string;
export type AlertEvery = number;
export type Timeout2 = number;
export type Webhook = WebhookConfig | WebhookConfig[] | null;
export type Url1 = string;
export type Method = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'HEAD';
export type Timeout3 = number | null;
export type Headers = {
	[k: string]: string;
} | null;
export type Body = string | null;
export type Confidence3 =
	| number
	| {
			[k: string]: number;
	  }
	| null;
export type CropPadding3 = number;
export type ExportRejected2 = boolean;
export type IncludeImage1 = boolean;
export type IncludePlot1 = boolean;
export type IncludeCrop1 = boolean;
export type IncludeVideo1 = boolean;
export type VideoWidth1 = number | null;
export type VideoCrf1 = number;
export type Token1 = string | null;
export type DataType = 'binary' | 'base64' | 'none';
export type DataMax = number | null;
export type Sse = SSEConfig | SSEConfig[] | null;
export type Confidence4 =
	| number
	| {
			[k: string]: number;
	  }
	| null;
export type CropPadding4 = number;
export type ExportRejected3 = boolean;
export type Port = number;
export type Endpoint = string | null;
export type Provider = string | null;
export type Winml = boolean;
export type Opset = number;
export type Url2 = string;
export type Method1 = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE' | 'HEAD';
export type Timeout4 = number;
export type Headers1 = {
	[k: string]: string;
} | null;
export type Body1 = string | null;
export type Interval1 = number;

export interface Config {
	detectors: Detectors;
	onnx?: OnnxConfig;
	health?: HealthcheckConfig | null;
}
export interface DetectorConfig {
	detection: DetectionConfig;
	yolo?: YoloConfig | null;
	identity?: IdentityConfig | null;
	vlm?: Vlm;
	exporters?: ExportersConfig | null;
}
export interface DetectionConfig {
	source: Source;
	interval?: Interval;
	frame_retention?: FrameRetention;
	frames_width?: FramesWidth;
}
export interface YoloConfig {
	model: Model;
	task?: Task;
	confidence?: Confidence;
	tracking?: Tracking;
	time_max?: TimeMax;
	timeout?: Timeout;
	cooldown?: Cooldown;
	include_trailing_time?: IncludeTrailingTime;
	frames_min?: FramesMin;
	imgsz?: Imgsz;
	iou?: Iou;
	tracker?: Tracker;
}
export interface IdentityConfig {
	database: Database;
	target_label: TargetLabel;
	display: IdentityDisplayConfig;
	candidate_filter: IdentityCandidateFilterConfig;
	controlled_zone: IdentityControlledZoneConfig;
	encoder: Encoder;
	similarity_threshold: SimilarityThreshold;
	similarity_margin: SimilarityMargin;
	query_frames: QueryFrames;
	gallery_frames: GalleryFrames;
	track_max_age: TrackMaxAge;
}
export interface IdentityDisplayConfig {
	singular: Singular;
	plural: Plural;
	official_id_label: OfficialIdLabel;
}
export interface IdentityCandidateFilterConfig {
	min_area_ratio: MinAreaRatio;
	max_area_ratio: MaxAreaRatio;
	frame_edge_margin: FrameEdgeMargin;
}
export interface IdentityControlledZoneConfig {
	zone_id: ZoneId;
	x1: X1;
	y1: Y1;
	x2: X2;
	y2: Y2;
	minimum_box_inside_ratio: MinimumBoxInsideRatio;
	minimum_stable_frames: MinimumStableFrames;
	clear_frames: ClearFrames;
}
export interface VLMConfig {
	prompt: Prompt;
	model: Model1;
	key?: Key;
	url?: Url;
	strategy?: Strategy;
	crop_padding?: CropPadding;
	timeout?: Timeout1;
}
export interface ExportersConfig {
	disk?: Disk;
	telegram?: Telegram;
	webhook?: Webhook;
	sse?: Sse;
}
export interface DiskConfig {
	confidence?: Confidence1;
	crop_padding?: CropPadding1;
	export_rejected?: ExportRejected;
	directory?: Directory;
	strategy?: Strategy1;
}
export interface ChatConfig {
	confidence?: Confidence2;
	crop_padding?: CropPadding2;
	export_rejected?: ExportRejected1;
	include_image?: IncludeImage;
	include_plot?: IncludePlot;
	include_crop?: IncludeCrop;
	include_video?: IncludeVideo;
	video_width?: VideoWidth;
	video_crf?: VideoCrf;
	token: Token;
	chat: Chat;
	alert_every?: AlertEvery;
	timeout?: Timeout2;
}
export interface WebhookConfig {
	url: Url1;
	method?: Method;
	timeout?: Timeout3;
	headers?: Headers;
	body?: Body;
	confidence?: Confidence3;
	crop_padding?: CropPadding3;
	export_rejected?: ExportRejected2;
	include_image?: IncludeImage1;
	include_plot?: IncludePlot1;
	include_crop?: IncludeCrop1;
	include_video?: IncludeVideo1;
	video_width?: VideoWidth1;
	video_crf?: VideoCrf1;
	token?: Token1;
	data_type?: DataType;
	data_max?: DataMax;
}
export interface SSEConfig {
	confidence?: Confidence4;
	crop_padding?: CropPadding4;
	export_rejected?: ExportRejected3;
	port?: Port;
	endpoint?: Endpoint;
}
export interface OnnxConfig {
	provider?: Provider;
	winml?: Winml;
	opset?: Opset;
}
export interface HealthcheckConfig {
	url: Url2;
	method?: Method1;
	timeout?: Timeout4;
	headers?: Headers1;
	body?: Body1;
	interval?: Interval1;
}
