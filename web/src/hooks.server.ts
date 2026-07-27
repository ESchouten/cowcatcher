await import('reflect-metadata');

if (typeof Reflect === 'undefined' || typeof Reflect.getMetadata !== 'function') {
	throw new Error('Failed to initialize reflect-metadata for server startup.');
}

export {};
