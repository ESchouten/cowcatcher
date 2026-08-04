declare module 'bun:sqlite' {
	export class Database {
		constructor(filename: string);
		close(): void;
		exec(sql: string): unknown;
		prepare(sql: string): {
			all(...bindings: unknown[]): unknown[];
			get(...bindings: unknown[]): unknown;
			run(...bindings: unknown[]): {
				changes: number | bigint;
				lastInsertRowid: number | bigint;
			};
		};
	}
}
