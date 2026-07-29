declare module 'onvif' {
	import type { EventEmitter } from 'node:events';
	import type { UrlWithStringQuery } from 'node:url';

	export interface CamOptions {
		hostname: string;
		username?: string;
		password?: string;
		port?: number | string | null;
		path?: string | null;
		timeout?: number;
		useSecure?: boolean;
		preserveAddress?: boolean;
	}

	export interface StreamUri {
		uri: string;
		invalidAfterConnect?: boolean;
		invalidAfterReboot?: boolean;
		timeout?: string;
	}

	export class Cam extends EventEmitter {
		constructor(options: CamOptions, callback?: (error?: Error | null) => void);
		hostname: string;
		port: number | string;
		path: string;
		urn?: string;
		xaddrs?: UrlWithStringQuery[];
		username?: string;
		password?: string;
		getStreamUri(
			options: { protocol: 'RTSP'; stream?: 'RTP-Unicast' | 'RTP-Multicast' },
			callback: (error: Error | null, stream?: StreamUri) => void
		): void;
	}

	export interface DiscoveryService extends EventEmitter {
		probe(
			options: { timeout: number },
			callback: (errors: Error[] | null, cameras: Cam[]) => void
		): void;
	}

	export const Discovery: DiscoveryService;

	const onvif: {
		Cam: typeof Cam;
		Discovery: DiscoveryService;
	};
	export default onvif;
}
