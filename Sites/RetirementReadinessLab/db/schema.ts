import { sqliteTable, text, integer, primaryKey, index } from 'drizzle-orm/sqlite-core';

export const metricReceipts = sqliteTable('metric_receipts', {
  id: text('id').primaryKey(),
  date: text('date').notNull(),
}, table => [index('metric_receipts_date').on(table.date)]);

export const metricDaily = sqliteTable('metric_daily', {
  date: text('date').notNull(),
  event: text('event').notNull(),
  source: text('source').notNull(),
  count: integer('count').notNull().default(0),
}, table => [primaryKey({ columns: [table.date, table.event, table.source] })]);

export const mcpUsage = sqliteTable('mcp_usage', {
  accountKey: text('account_key').primaryKey(),
  hour: integer('hour').notNull(), calls: integer('calls').notNull(),
  lease: text('lease').notNull(), leaseUntil: integer('lease_until').notNull(),
  expiresAt: integer('expires_at').notNull(),
}, table => [index('mcp_usage_expiry').on(table.expiresAt)]);

export const mcpDaily = sqliteTable('mcp_daily', {
  date: text('date').notNull(), tool: text('tool').notNull(),
  outcome: text('outcome').notNull(), durationBand: text('duration_band').notNull(),
  count: integer('count').notNull().default(0),
}, table => [primaryKey({ columns: [table.date, table.tool, table.outcome, table.durationBand] })]);

// Temporary control state only; no raw identities, request IDs or scenarios.
export const mcpExecution = sqliteTable('mcp_execution', {
  accountKey: text('account_key').primaryKey(), lease: text('lease').notNull(),
  requestKey: text('request_key').notNull(), cancelled: integer('cancelled').notNull().default(0),
  expiresAt: integer('expires_at').notNull(),
}, table => [index('mcp_execution_expiry').on(table.expiresAt)]);
