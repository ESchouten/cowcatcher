import { compileFromFile } from 'json-schema-to-typescript';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const webDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const schemaDirectory = resolve(webDirectory, '../config');
const outputDirectory = resolve(webDirectory, 'src/lib/generated');
const options = {
	bannerComment: '/* Generated from config JSON Schema. Do not edit manually. */',
	style: {
		printWidth: 100,
		singleQuote: true,
		tabWidth: 2,
		useTabs: true
	}
};

await mkdir(outputDirectory, { recursive: true });

for (const [input, output] of [
	['config.schema.json', 'config.ts'],
	['metadata.schema.json', 'metadata.ts']
]) {
	const source = await compileFromFile(resolve(schemaDirectory, input), options);
	await writeFile(resolve(outputDirectory, output), source);
}
