/* Generated from config JSON Schema. Do not edit manually. */

export type EventId = string;
export type Source = string;
export type Timestamp = string;
export type Validated = boolean | null;
export type Confidence = number;
export type Status = 'matched' | 'unknown' | 'ambiguous' | 'insufficient_evidence' | 'switch_risk';
export type VisualIdentityId = string | null;
export type OfficialId = string | null;
export type Similarity = number | null;
export type Margin = number | null;
export type IdentityResults = IdentityMetadata[];
export type Observations = number;
export type Start = string;
export type End = string;
export type Duration = number;
export type X1 = number;
export type Y1 = number;
export type X2 = number;
export type Y2 = number;
export type Label = string | null;
export type Confidence1 = number | null;
export type TrackId = number | null;
export type Crops = CropMetadata[];

export interface DetectionMetadata {
	event_id: EventId;
	source: Source;
	timestamp: Timestamp;
	validated: Validated;
	confidence: Confidence;
	confidences: Confidences;
	identity_results: IdentityResults;
	observations: Observations;
	start: Start;
	end: End;
	duration: Duration;
	crop?: CropMetadata | null;
	crops?: Crops;
}
export interface Confidences {
	[k: string]: number;
}
export interface IdentityMetadata {
	status: Status;
	visual_identity_id: VisualIdentityId;
	official_id: OfficialId;
	similarity: Similarity;
	margin: Margin;
}
export interface CropMetadata {
	x1: X1;
	y1: Y1;
	x2: X2;
	y2: Y2;
	label?: Label;
	confidence?: Confidence1;
	track_id?: TrackId;
	identity?: IdentityMetadata | null;
}
