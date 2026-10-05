CREATE TABLE `mcp_daily` (
	`date` text NOT NULL,
	`tool` text NOT NULL,
	`outcome` text NOT NULL,
	`duration_band` text NOT NULL,
	`count` integer DEFAULT 0 NOT NULL,
	PRIMARY KEY(`date`, `tool`, `outcome`, `duration_band`)
);
--> statement-breakpoint
CREATE TABLE `mcp_usage` (
	`account_key` text PRIMARY KEY NOT NULL,
	`hour` integer NOT NULL,
	`calls` integer NOT NULL,
	`lease` text NOT NULL,
	`lease_until` integer NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `mcp_usage_expiry` ON `mcp_usage` (`expires_at`);