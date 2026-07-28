import configSchema from '$lib/generated/config.schema.json';
import Ajv, { type ValidateFunction } from 'ajv';

export type PresetCategory = 'detector' | 'identity' | 'vlm';
export interface PresetDocument {
	filename: string;
	blobSha: string;
	value: unknown;
}

const schemaName: Record<PresetCategory, string> = {
	detector: 'DetectorConfig',
	identity: 'IdentityConfig',
	vlm: 'VLMConfig'
};
const ajv = new Ajv({ allErrors: true, strict: false });
const fullSchema = configSchema as Record<string, unknown>;
const partialSchema = withoutRequired(structuredClone(fullSchema)) as Record<string, unknown>;
const validators = Object.fromEntries(
	Object.entries(schemaName).map(([category, name]) => [
		category,
		ajv.compile({
			$defs: (category === 'detector' ? partialSchema : fullSchema).$defs,
			$ref: `#/$defs/${name}`
		})
	])
) as Record<PresetCategory, ValidateFunction>;
const detectorValidator = ajv.compile({
	$defs: fullSchema.$defs,
	$ref: '#/$defs/DetectorConfig'
});

export function validatePreset(category: PresetCategory, value: unknown): void {
	validate(validators[category], value, `${category} preset`);
	if (category === 'identity') validateIdentity(value);
}

export function decodePresetBlob(
	category: PresetCategory,
	filename: string,
	blobSha: string,
	base64Content: string
): PresetDocument {
	let value: unknown;
	try {
		value = JSON.parse(Buffer.from(base64Content.replace(/\s/g, ''), 'base64').toString('utf8'));
	} catch (error) {
		throw new Error(`Invalid JSON in ${category} preset '${filename}'`, { cause: error });
	}
	validatePreset(category, value);
	return { filename, blobSha, value };
}

export function validateResolvedDetector(value: unknown): void {
	validate(detectorValidator, value, 'detector configuration');
	const detector = value as {
		yolo?: { tracking?: boolean; confidence?: number | Record<string, number> };
		identity?: { target_label: string } & Record<string, unknown>;
	};
	if (!detector.identity) return;
	validateIdentity(detector.identity);
	if (!detector.yolo?.tracking) {
		throw new Error('Identity requires the selected detector to enable tracking');
	}
	if (
		typeof detector.yolo.confidence === 'object' &&
		!(detector.identity.target_label in detector.yolo.confidence)
	) {
		throw new Error(
			`Identity target '${detector.identity.target_label}' is not enabled by detector confidence`
		);
	}
}

function validate(validator: ValidateFunction, value: unknown, label: string): void {
	if (!validator(value)) {
		throw new Error(`Invalid ${label}: ${ajv.errorsText(validator.errors)}`);
	}
}

function validateIdentity(value: unknown): void {
	const identity = value as { database?: string };
	if (identity.database && isAbsolutePath(identity.database)) {
		throw new Error('Invalid identity preset: database path must be relative to config.json');
	}
}

function withoutRequired(value: unknown): unknown {
	if (Array.isArray(value)) {
		return value.map(withoutRequired);
	}
	if (!value || typeof value !== 'object') {
		return value;
	}
	return Object.fromEntries(
		Object.entries(value)
			.filter(([key]) => key !== 'required')
			.map(([key, child]) => [key, withoutRequired(child)])
	);
}

function isAbsolutePath(value: string): boolean {
	return value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value) || value.startsWith('\\\\');
}
