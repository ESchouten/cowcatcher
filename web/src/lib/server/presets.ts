import configSchema from '$lib/generated/config.schema.json';
import { createHash } from 'node:crypto';

export type PresetCategory = 'detector' | 'identity' | 'vlm';
export interface PresetDocument {
	filename: string;
	blobSha: string;
	value: unknown;
}

type JsonSchema = {
	$ref?: string;
	anyOf?: JsonSchema[];
	oneOf?: JsonSchema[];
	allOf?: JsonSchema[];
	type?: string | string[];
	const?: unknown;
	enum?: unknown[];
	properties?: Record<string, JsonSchema>;
	required?: string[];
	additionalProperties?: boolean | JsonSchema;
	items?: JsonSchema;
	minItems?: number;
	maxItems?: number;
	minLength?: number;
	maxLength?: number;
	pattern?: string;
	minimum?: number;
	maximum?: number;
	exclusiveMinimum?: number;
	exclusiveMaximum?: number;
};

type ConfigSchema = JsonSchema & {
	$defs: Record<string, JsonSchema>;
};

const localSchema = configSchema as ConfigSchema;
const schemaName: Record<PresetCategory, string> = {
	detector: 'DetectorConfig',
	identity: 'IdentityConfig',
	vlm: 'VLMConfig'
};

export function validatePreset(category: PresetCategory, value: unknown): void {
	const schema = localSchema.$defs[schemaName[category]];
	if (!schema) {
		throw new Error(`Local configuration schema has no ${schemaName[category]} definition`);
	}
	const errors: string[] = [];
	validate(value, schema, '$', errors, category === 'detector');
	if (errors.length) {
		throw new Error(`Invalid ${category} preset:\n${errors.join('\n')}`);
	}
	if (category === 'identity') {
		validateIdentitySemantics(value as Record<string, unknown>);
	}
}

export function decodePresetBlob(
	category: PresetCategory,
	filename: string,
	blobSha: string,
	base64Content: string
): PresetDocument {
	const bytes = Buffer.from(base64Content.replace(/\s/g, ''), 'base64');
	const expectedBlobSha = createHash('sha1')
		.update(`blob ${bytes.byteLength}\0`)
		.update(bytes)
		.digest('hex');
	if (expectedBlobSha !== blobSha) {
		throw new Error(`GitHub blob SHA mismatch for ${category} preset '${filename}'`);
	}
	let value: unknown;
	try {
		value = JSON.parse(bytes.toString('utf8'));
	} catch (error) {
		throw new Error(`Invalid JSON in ${category} preset '${filename}'`, { cause: error });
	}
	validatePreset(category, value);
	return { filename, blobSha, value };
}

export function validateResolvedDetector(value: unknown): void {
	const schema = localSchema.$defs.DetectorConfig;
	const errors: string[] = [];
	validate(value, schema, '$', errors, false);
	if (errors.length) {
		throw new Error(`Invalid resolved detector configuration:\n${errors.join('\n')}`);
	}
	const detector = value as {
		yolo?: { tracking?: boolean; confidence?: number | Record<string, number> };
		identity?: { target_label: string } & Record<string, unknown>;
	};
	if (!detector.identity) return;
	validateIdentitySemantics(detector.identity);
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

function validate(
	value: unknown,
	schema: JsonSchema,
	path: string,
	errors: string[],
	partial: boolean
): void {
	if (schema.$ref) {
		const resolved = resolveReference(schema.$ref);
		if (!resolved) {
			errors.push(`${path}: unresolved local schema reference ${schema.$ref}`);
			return;
		}
		validate(value, resolved, path, errors, partial);
		return;
	}

	if (schema.allOf) {
		for (const member of schema.allOf) validate(value, member, path, errors, partial);
	}
	const union = schema.anyOf ?? schema.oneOf;
	if (union) {
		const variants = union.map((member) => {
			const variantErrors: string[] = [];
			validate(value, member, path, variantErrors, partial);
			return variantErrors;
		});
		if (variants.every((variant) => variant.length > 0)) {
			errors.push(`${path}: value does not match an allowed schema`);
		}
		return;
	}

	if ('const' in schema && value !== schema.const) {
		errors.push(`${path}: must equal ${JSON.stringify(schema.const)}`);
		return;
	}
	if (schema.enum && !schema.enum.some((item) => Object.is(item, value))) {
		errors.push(`${path}: is not an allowed value`);
		return;
	}

	const types = Array.isArray(schema.type) ? schema.type : schema.type ? [schema.type] : [];
	if (types.length && !types.some((type) => matchesType(value, type))) {
		errors.push(`${path}: expected ${types.join(' or ')}`);
		return;
	}

	if (typeof value === 'string') {
		if (schema.minLength !== undefined && value.length < schema.minLength) {
			errors.push(`${path}: string is too short`);
		}
		if (schema.maxLength !== undefined && value.length > schema.maxLength) {
			errors.push(`${path}: string is too long`);
		}
		if (schema.pattern && !new RegExp(schema.pattern).test(value)) {
			errors.push(`${path}: string does not match the required pattern`);
		}
	}

	if (typeof value === 'number') {
		if (schema.minimum !== undefined && value < schema.minimum) {
			errors.push(`${path}: must be at least ${schema.minimum}`);
		}
		if (schema.maximum !== undefined && value > schema.maximum) {
			errors.push(`${path}: must be at most ${schema.maximum}`);
		}
		if (schema.exclusiveMinimum !== undefined && value <= schema.exclusiveMinimum) {
			errors.push(`${path}: must be greater than ${schema.exclusiveMinimum}`);
		}
		if (schema.exclusiveMaximum !== undefined && value >= schema.exclusiveMaximum) {
			errors.push(`${path}: must be less than ${schema.exclusiveMaximum}`);
		}
	}

	if (Array.isArray(value)) {
		if (schema.minItems !== undefined && value.length < schema.minItems) {
			errors.push(`${path}: array has too few items`);
		}
		if (schema.maxItems !== undefined && value.length > schema.maxItems) {
			errors.push(`${path}: array has too many items`);
		}
		if (schema.items) {
			value.forEach((item, index) =>
				validate(item, schema.items as JsonSchema, `${path}[${index}]`, errors, partial)
			);
		}
	}

	if (isObject(value)) {
		if (!partial) {
			for (const name of schema.required ?? []) {
				if (!(name in value)) errors.push(`${path}.${name}: required property is missing`);
			}
		}
		for (const [name, propertyValue] of Object.entries(value)) {
			const propertySchema = schema.properties?.[name];
			if (propertySchema) {
				validate(propertyValue, propertySchema, `${path}.${name}`, errors, partial);
			} else if (schema.additionalProperties === false) {
				errors.push(`${path}.${name}: unknown property`);
			} else if (isObject(schema.additionalProperties)) {
				validate(propertyValue, schema.additionalProperties, `${path}.${name}`, errors, partial);
			}
		}
	}
}

function resolveReference(reference: string): JsonSchema | undefined {
	const prefix = '#/$defs/';
	return reference.startsWith(prefix)
		? localSchema.$defs[reference.slice(prefix.length)]
		: undefined;
}

function matchesType(value: unknown, type: string): boolean {
	switch (type) {
		case 'null':
			return value === null;
		case 'object':
			return isObject(value);
		case 'array':
			return Array.isArray(value);
		case 'integer':
			return typeof value === 'number' && Number.isInteger(value);
		default:
			return typeof value === type;
	}
}

function isObject(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function validateIdentitySemantics(value: Record<string, unknown>): void {
	const candidate = value.candidate_filter as
		| { min_area_ratio?: number; max_area_ratio?: number }
		| undefined;
	if (
		candidate?.min_area_ratio !== undefined &&
		candidate.max_area_ratio !== undefined &&
		candidate.min_area_ratio > candidate.max_area_ratio
	) {
		throw new Error('Invalid identity preset: minimum area exceeds maximum area');
	}
	const zone = value.controlled_zone as
		| { x1?: number; y1?: number; x2?: number; y2?: number }
		| undefined;
	if (
		zone?.x1 !== undefined &&
		zone.y1 !== undefined &&
		zone.x2 !== undefined &&
		zone.y2 !== undefined &&
		(zone.x2 <= zone.x1 || zone.y2 <= zone.y1)
	) {
		throw new Error('Invalid identity preset: controlled zone must have positive extent');
	}
	if (value.encoder === 'miewid-dual-crop-v1') {
		if (value.query_frames !== 2 || value.gallery_frames !== 4) {
			throw new Error(
				'Invalid identity preset: miewid-dual-crop-v1 requires two query and four gallery frames'
			);
		}
	}
	if (typeof value.database === 'string' && isAbsolutePath(value.database)) {
		throw new Error('Invalid identity preset: database path must be relative to config.json');
	}
}

function isAbsolutePath(value: string): boolean {
	return value.startsWith('/') || /^[A-Za-z]:[\\/]/.test(value) || value.startsWith('\\\\');
}
