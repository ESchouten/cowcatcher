declare const __AI_DETECTOR_WEB_TARGET__: string;

export interface SqliteRunResult {
	changes: number | bigint;
	lastInsertRowid: number | bigint;
}

export interface SqliteStatement {
	all(...bindings: unknown[]): unknown[];
	get(...bindings: unknown[]): unknown;
	run(...bindings: unknown[]): SqliteRunResult;
}

export interface SqliteDatabase {
	close(): void;
	exec(sql: string): unknown;
	prepare(sql: string): SqliteStatement;
}

export async function openSqliteDatabase(filename: string): Promise<SqliteDatabase> {
	if (__AI_DETECTOR_WEB_TARGET__ === 'node') {
		const { DatabaseSync } = await import('node:sqlite');
		return new DatabaseSync(filename) as unknown as SqliteDatabase;
	}

	const { Database } = await import('bun:sqlite');
	return new Database(filename) as unknown as SqliteDatabase;
}
