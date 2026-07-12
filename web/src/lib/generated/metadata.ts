/* Generated from config JSON Schema. Do not edit manually. */

export type Timestamp = string;
export type Validated = boolean | null;
export type Confidence = number;
export type Detections = number;
export type Start = string;
export type End = string;
export type Duration = number;
export type X1 = number;
export type Y1 = number;
export type X2 = number;
export type Y2 = number;
export type Label = string | null;
export type Confidence1 = number | null;
export type Crops = CropMetadata[];

export interface DetectionMetadata {
	timestamp: Timestamp;
	validated: Validated;
	confidence: Confidence;
	confidences: Confidences;
	detections: Detections;
	start: Start;
	end: End;
	duration: Duration;
	crop?: CropMetadata | null;
	crops?: Crops;
}
export interface Confidences {
	[k: string]: number;
}
export interface CropMetadata {
	x1: X1;
	y1: Y1;
	x2: X2;
	y2: Y2;
	label?: Label;
	confidence?: Confidence1;
}
