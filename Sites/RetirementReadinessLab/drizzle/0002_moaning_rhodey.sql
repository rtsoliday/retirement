CREATE TABLE `mcp_execution` (
	`account_key` text PRIMARY KEY NOT NULL,
	`lease` text NOT NULL,
	`request_key` text NOT NULL,
	`cancelled` integer DEFAULT 0 NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `mcp_execution_expiry` ON `mcp_execution` (`expires_at`);