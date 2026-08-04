export function redactCameraSource(source: string): string {
	try {
		const url = new URL(source);
		if (!['rtsp:', 'rtsps:'].includes(url.protocol)) return 'Camera stream';
		return `${url.protocol}//${url.host}${url.pathname}`;
	} catch {
		return 'Camera stream';
	}
}
