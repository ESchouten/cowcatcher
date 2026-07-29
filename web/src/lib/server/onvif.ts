import { randomUUID } from 'node:crypto';
import onvif, { type Cam as OnvifCam, type CamOptions } from 'onvif';

const { Cam, Discovery } = onvif;

const MIN_DISCOVERY_TIMEOUT_MS = 1_000;
const MAX_DISCOVERY_TIMEOUT_MS = 10_000;
const CAMERA_CONNECT_TIMEOUT_MS = 12_000;
const DISCOVERY_TOKEN_TTL_MS = 10 * 60 * 1_000;

export interface DiscoveredOnvifCamera {
	id: string;
	name: string;
	host: string;
	port: number;
	secure: boolean;
}

export interface ResolvedOnvifStream {
	label: string;
	source: string;
}

interface CameraEndpoint {
	expiresAt: number;
	host: string;
	port: number;
	path: string;
	secure: boolean;
}

interface DiscoveryCamera {
	hostname: string;
	port: number | string;
	path?: string;
	urn?: string;
	xaddrs?: Array<{ protocol?: string | null }>;
}

const cameraEndpoints = new Map<string, CameraEndpoint>();
let discoveryTail: Promise<void> = Promise.resolve();

function removeExpiredEndpoints(now = Date.now()): void {
	for (const [id, endpoint] of cameraEndpoints) {
		if (endpoint.expiresAt <= now) cameraEndpoints.delete(id);
	}
}

function normalizePort(value: number | string, secure: boolean): number {
	const port = Number(value || (secure ? 443 : 80));
	if (!Number.isInteger(port) || port < 1 || port > 65_535) {
		throw new Error('ONVIF camera returned an invalid port');
	}
	return port;
}

export function describeDiscoveredCamera(
	camera: DiscoveryCamera,
	now = Date.now()
): DiscoveredOnvifCamera {
	const secure = camera.xaddrs?.some((address) => address.protocol === 'https:') ?? false;
	const host = camera.hostname.trim();
	if (!host) throw new Error('ONVIF camera returned an empty hostname');
	const port = normalizePort(camera.port, secure);
	const id = randomUUID();
	cameraEndpoints.set(id, {
		expiresAt: now + DISCOVERY_TOKEN_TTL_MS,
		host,
		port,
		path: camera.path || '/onvif/device_service',
		secure
	});
	return {
		id,
		name: `Camera at ${host}`,
		host,
		port,
		secure
	};
}

function probe(timeoutMs: number): Promise<DiscoveredOnvifCamera[]> {
	return new Promise((resolve, reject) => {
		const discoveryError = () => {
			// Malformed replies from unrelated WS-Discovery devices are non-fatal.
		};
		Discovery.on('error', discoveryError);
		Discovery.probe({ timeout: timeoutMs }, (errors, cameras) => {
			Discovery.off('error', discoveryError);
			try {
				const discovered = cameras.map((camera) =>
					describeDiscoveredCamera(camera as DiscoveryCamera)
				);
				if (!discovered.length && errors?.length) {
					reject(new Error('Camera discovery received only invalid ONVIF replies'));
					return;
				}
				resolve(
					discovered.sort(
						(first, second) =>
							first.host.localeCompare(second.host, undefined, { numeric: true }) ||
							first.port - second.port
					)
				);
			} catch (error) {
				reject(error);
			}
		});
	});
}

export function discoverOnvifCameras(timeoutMs = 4_000): Promise<DiscoveredOnvifCamera[]> {
	const boundedTimeout = Math.max(
		MIN_DISCOVERY_TIMEOUT_MS,
		Math.min(MAX_DISCOVERY_TIMEOUT_MS, timeoutMs)
	);
	removeExpiredEndpoints();
	const result = discoveryTail.then(() => probe(boundedTimeout));
	discoveryTail = result.then(
		() => undefined,
		() => undefined
	);
	return result;
}

function connectCamera(options: CamOptions): Promise<OnvifCam> {
	return new Promise((resolve, reject) => {
		const camera = new Cam(options, (error) => {
			if (error) {
				reject(new Error(`Could not authenticate with the ONVIF camera: ${error.message}`));
				return;
			}
			resolve(camera);
		});
	});
}

function getStreamUri(camera: OnvifCam): Promise<string> {
	return new Promise((resolve, reject) => {
		camera.getStreamUri({ protocol: 'RTSP', stream: 'RTP-Unicast' }, (error, stream) => {
			if (error) {
				reject(new Error(`The camera did not provide an RTSP stream: ${error.message}`));
				return;
			}
			if (!stream?.uri) {
				reject(new Error('The camera returned an empty RTSP stream address'));
				return;
			}
			resolve(stream.uri);
		});
	});
}

export function addCredentialsToStream(
	streamUri: string,
	host: string,
	username: string,
	password: string
): string {
	const stream = new URL(streamUri);
	if (stream.protocol !== 'rtsp:' && stream.protocol !== 'rtsps:') {
		throw new Error(`Unsupported camera stream protocol: ${stream.protocol}`);
	}
	if (['0.0.0.0', '127.0.0.1', 'localhost'].includes(stream.hostname)) {
		stream.hostname = host;
	}
	if (username) {
		stream.username = username;
		stream.password = password;
	}
	return stream.toString();
}

export async function resolveOnvifStream(
	id: string,
	username: string,
	password: string
): Promise<ResolvedOnvifStream> {
	removeExpiredEndpoints();
	const endpoint = cameraEndpoints.get(id);
	if (!endpoint) {
		throw new Error('This camera discovery result expired; scan the network again');
	}
	const camera = await connectCamera({
		hostname: endpoint.host,
		port: endpoint.port,
		path: endpoint.path,
		useSecure: endpoint.secure,
		username,
		password,
		timeout: CAMERA_CONNECT_TIMEOUT_MS,
		preserveAddress: true
	});
	const source = addCredentialsToStream(
		await getStreamUri(camera),
		endpoint.host,
		username,
		password
	);
	return {
		label: `Camera ${endpoint.host}`,
		source
	};
}
