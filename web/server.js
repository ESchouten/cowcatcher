import http from 'node:http';
import process from 'node:process';

process.env.PROTOCOL_HEADER ??= 'x-forwarded-proto';
process.env.HOST_HEADER ??= 'x-forwarded-host';

const { handler } = await import('./build/handler.js');
const host = process.env.HOST ?? '0.0.0.0';
const port = Number(process.env.PORT ?? 3000);
const shutdownTimeout = Number(process.env.SHUTDOWN_TIMEOUT ?? 30) * 1000;

const server = http.createServer((request, response) => {
	request.headers['x-forwarded-proto'] ||= 'http';
	request.headers['x-forwarded-host'] ||= request.headers.host;
	handler(request, response);
});

server.listen(port, host, () => {
	console.log(`Listening on http://${host}:${port}`);
});

let closing = false;
function shutdown() {
	if (closing) return;
	closing = true;
	server.closeIdleConnections();
	server.close();
	setTimeout(() => server.closeAllConnections(), shutdownTimeout).unref();
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
