CREATE TABLE `metric_daily` (
	`date` text NOT NULL,
	`event` text NOT NULL,
	`source` text NOT NULL,
	`count` integer DEFAULT 0 NOT NULL,
	PRIMARY KEY(`date`, `event`, `source`)
);
--> statement-breakpoint
CREATE TABLE `metric_receipts` (
	`id` text PRIMARY KEY NOT NULL,
	`date` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `metric_receipts_date` ON `metric_receipts` (`date`);