#!/usr/bin/env python3
"""Rank public Fidelity funds by return and volatility.

By default, the script downloads the current open Fidelity-managed retail
mutual fund universe from Fidelity's public fund screener API. It then fetches
adjusted daily price history from Yahoo Finance's chart endpoint, uses the most
recent 10 years when available, and otherwise uses the longest history returned
for each fund. It ranks funds by this score:

Score = AverageAnnReturn - abs((AverageAnnReturn * 0.45) - AnnStdDev)

That favors high-return funds whose annualized standard deviation is close to
45% of annualized return.

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import html
import json
import math
import re
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Sequence


SAMPLE_TICKERS = ("FFTHX", "FFFFX", "FFFEX", "FFTWX", "FFFHX", "FFFGX")
DAYS_PER_YEAR = 365.2425
TRADING_DAYS_PER_YEAR = 252
TARGET_STD_DEV_TO_RETURN_RATIO = 0.45
FIDELITY_FUND_SCREENER_URL = (
    "https://fundresearch.fidelity.com/fund-screener/api/search/v1/funds"
)
FIDELITY_ROWS_PER_PAGE = 1000
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HTML_TAG_RE = re.compile(r"<[^>]+>")
VALID_TICKER_RE = re.compile(r"^[A-Z0-9.\-]{1,12}$")


@dataclass(frozen=True)
class FundListing:
    ticker: str
    name: str = ""
    status: str = ""
    asset_class: str = ""
    category: str = ""
    universe_as_of_date: str = ""


@dataclass(frozen=True)
class PricePoint:
    date: date
    adjusted_close: float


@dataclass
class FundMetrics:
    ticker: str
    start_date: date
    end_date: date
    years_used: float
    data_points: int
    annualized_return: float
    annualized_std_dev: float
    name: str = ""
    status: str = ""
    asset_class: str = ""
    category: str = ""
    universe_as_of_date: str = ""
    target_std_dev: float = 0.0
    std_dev_gap: float = 0.0
    score: float = 0.0
    final_rank: int = 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank public Fidelity funds using recent adjusted-close price "
            "history. By default, the fund universe is downloaded from "
            "Fidelity's public screener. Higher score is better."
        )
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        help=(
            "Specific tickers to rank. If omitted, downloads all open "
            "Fidelity-managed retail mutual funds from Fidelity."
        ),
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help=f"Rank only the original sample tickers: {' '.join(SAMPLE_TICKERS)}.",
    )
    parser.add_argument(
        "--include-closed",
        action="store_true",
        help="Include Fidelity funds that are not open to new investors.",
    )
    parser.add_argument(
        "--include-etfs",
        action="store_true",
        help="Also request Fidelity ETFs from the Fidelity screener API.",
    )
    parser.add_argument(
        "--lookback-years",
        type=int,
        default=10,
        help="Years of history to use when available. Default: 10.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout per fund in seconds. Default: 30.",
    )
    parser.add_argument(
        "--fund-list-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout for each Fidelity fund-list request. Default: 30.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent Yahoo Finance downloads. Default: 8.",
    )
    parser.add_argument(
        "--min-years",
        type=float,
        default=0.0,
        help="Skip funds with less than this many years of usable history.",
    )
    parser.add_argument(
        "--top",
        type=int,
        help="Only print the top N funds. CSV and JSON still include all ranked funds.",
    )
    parser.add_argument(
        "--show-skipped",
        action="store_true",
        help="Print every skipped fund and error instead of a short summary.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code if any fund is skipped.",
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        help="Optional path to also write the ranked results as CSV.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of the human-readable table.",
    )
    return parser.parse_args(argv)


def clean_fund_name(value: object) -> str:
    if value is None:
        return ""
    without_tags = HTML_TAG_RE.sub("", str(value))
    return " ".join(html.unescape(without_tags).split())


def http_json_request(
    url: str,
    timeout: float,
    *,
    label: str,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    data = None
    headers = {
        "Accept": "application/json",
        "User-Agent": "retirement-fund-ranker/1.0",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{label}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{label}: could not connect: {exc.reason}") from exc
    except TimeoutError as exc:
        raise RuntimeError(f"{label}: request timed out") from exc
    except OSError as exc:
        raise RuntimeError(f"{label}: network error: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label}: returned invalid JSON") from exc


def parse_fidelity_listing(
    raw_fund: dict[str, object], universe_as_of_date: str
) -> FundListing | None:
    raw_info = raw_fund.get("fundInformation")
    if not isinstance(raw_info, dict):
        return None

    ticker = str(raw_info.get("ticker") or raw_info.get("tradingSymbol") or "")
    ticker = ticker.strip().upper()
    if not ticker or not VALID_TICKER_RE.fullmatch(ticker):
        return None

    return FundListing(
        ticker=ticker,
        name=clean_fund_name(raw_info.get("legalName") or raw_info.get("portfolioName")),
        status=clean_fund_name(raw_info.get("statusDescription")),
        asset_class=clean_fund_name(raw_info.get("mstarAssetClassName")),
        category=clean_fund_name(raw_info.get("mstarCategoryName")),
        universe_as_of_date=universe_as_of_date,
    )


def fetch_fidelity_public_funds(
    timeout: float, *, include_closed: bool, include_etfs: bool
) -> list[FundListing]:
    listings_by_ticker: dict[str, FundListing] = {}
    investment_type = "MFN,ETF" if include_etfs else "MFN"
    open_to_new_investors = "NEW,CLOSED" if include_closed else "OPEN"
    page_number = 1

    while True:
        payload: dict[str, object] = {
            "businessChannel": "RETAIL",
            "currentPageNumber": page_number,
            "noOfRowsPerPage": FIDELITY_ROWS_PER_PAGE,
            "sortBy": "legalName",
            "sortOrder": "ASC",
            "subjectAreaCode": "fundInformation",
            "searchFilter": {
                "fidelityFundOnly": "F",
                "includeLeveragedAndInverseFunds": "N",
                "investmentTypeCode": investment_type,
                "openToNewInvestors": open_to_new_investors,
            },
        }
        data = http_json_request(
            FIDELITY_FUND_SCREENER_URL,
            timeout,
            label="Fidelity fund screener",
            payload=payload,
        )
        raw_funds = data.get("funds") or []
        if not isinstance(raw_funds, list):
            raise RuntimeError("Fidelity fund screener: unexpected response format")

        universe_as_of_date = clean_fund_name(data.get("asOfDate"))
        for raw_fund in raw_funds:
            if not isinstance(raw_fund, dict):
                continue
            listing = parse_fidelity_listing(raw_fund, universe_as_of_date)
            if listing is not None:
                listings_by_ticker[listing.ticker] = listing

        if len(raw_funds) < FIDELITY_ROWS_PER_PAGE:
            break
        page_number += 1
        if page_number > 100:
            raise RuntimeError("Fidelity fund screener: pagination did not terminate")

    return sorted(listings_by_ticker.values(), key=lambda listing: listing.ticker)


def listings_from_tickers(tickers: Sequence[str]) -> list[FundListing]:
    listings: list[FundListing] = []
    seen: set[str] = set()
    for ticker in tickers:
        normalized = ticker.strip().upper()
        if not normalized or normalized in seen:
            continue
        listings.append(FundListing(ticker=normalized))
        seen.add(normalized)
    return listings


def subtract_years(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        # Handles Feb. 29 when the target year is not a leap year.
        return value.replace(month=2, day=28, year=value.year - years)


def midnight_utc_timestamp(value: date) -> int:
    return int(datetime(value.year, value.month, value.day, tzinfo=timezone.utc).timestamp())


def fetch_adjusted_history(
    ticker: str, timeout: float, lookback_years: int
) -> list[PricePoint]:
    encoded_ticker = urllib.parse.quote(ticker.upper())
    # Ask for a small buffer before the lookback window so the final selection
    # can be based on the latest available market close, not today's date.
    now = datetime.now(timezone.utc)
    today = now.date()
    period1 = subtract_years(today, lookback_years) - timedelta(days=14)
    url = (
        YAHOO_CHART_URL.format(ticker=encoded_ticker)
        + f"?period1={midnight_utc_timestamp(period1)}"
        + f"&period2={int(now.timestamp())}"
        + "&interval=1d&events=history&includeAdjustedClose=true"
    )
    payload = http_json_request(url, timeout, label=f"{ticker}: Yahoo Finance")

    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description") or error.get("code") or "unknown error"
        raise RuntimeError(f"{ticker}: Yahoo Finance error: {description}")

    results = chart.get("result") or []
    if not results:
        raise RuntimeError(f"{ticker}: Yahoo Finance returned no chart data")

    result = results[0]
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    adjclose_sets = indicators.get("adjclose") or []
    quote_sets = indicators.get("quote") or []

    adjusted_closes: list[float | None] = []
    if adjclose_sets and adjclose_sets[0].get("adjclose"):
        adjusted_closes = adjclose_sets[0]["adjclose"]
    elif quote_sets and quote_sets[0].get("close"):
        adjusted_closes = quote_sets[0]["close"]

    if not timestamps or not adjusted_closes:
        raise RuntimeError(f"{ticker}: Yahoo Finance returned no adjusted prices")

    points: list[PricePoint] = []
    for timestamp, adjusted_close in zip(timestamps, adjusted_closes):
        if adjusted_close is None or adjusted_close <= 0:
            continue
        point_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).date()
        points.append(PricePoint(point_date, float(adjusted_close)))

    points.sort(key=lambda point: point.date)
    deduped: list[PricePoint] = []
    for point in points:
        if deduped and deduped[-1].date == point.date:
            deduped[-1] = point
        else:
            deduped.append(point)

    if len(deduped) < 3:
        raise RuntimeError(f"{ticker}: not enough price history to calculate volatility")

    return deduped


def select_lookback(points: Sequence[PricePoint], lookback_years: int) -> list[PricePoint]:
    latest_date = points[-1].date
    target_start = subtract_years(latest_date, lookback_years)

    if points[0].date <= target_start:
        selected = [point for point in points if point.date >= target_start]
    else:
        selected = list(points)

    if len(selected) < 3:
        raise RuntimeError("not enough price history in selected timeframe")
    return selected


def calculate_metrics(
    listing: FundListing, points: Sequence[PricePoint], lookback_years: int
) -> FundMetrics:
    selected = select_lookback(points, lookback_years)
    start = selected[0]
    end = selected[-1]
    years_used = (end.date - start.date).days / DAYS_PER_YEAR
    if years_used <= 0:
        raise RuntimeError(f"{listing.ticker}: selected timeframe is too short")

    daily_returns = [
        current.adjusted_close / previous.adjusted_close - 1.0
        for previous, current in zip(selected, selected[1:])
    ]
    if len(daily_returns) < 2:
        raise RuntimeError(f"{listing.ticker}: not enough returns to calculate volatility")

    annualized_return = (end.adjusted_close / start.adjusted_close) ** (
        1.0 / years_used
    ) - 1.0
    annualized_std_dev = statistics.stdev(daily_returns) * math.sqrt(
        TRADING_DAYS_PER_YEAR
    )

    return FundMetrics(
        ticker=listing.ticker,
        start_date=start.date,
        end_date=end.date,
        years_used=years_used,
        data_points=len(selected),
        annualized_return=annualized_return,
        annualized_std_dev=annualized_std_dev,
        name=listing.name,
        status=listing.status,
        asset_class=listing.asset_class,
        category=listing.category,
        universe_as_of_date=listing.universe_as_of_date,
    )


def calculate_listing_metrics(
    listing: FundListing, lookback_years: int, timeout: float
) -> FundMetrics:
    history = fetch_adjusted_history(
        listing.ticker, timeout=timeout, lookback_years=lookback_years
    )
    return calculate_metrics(listing, history, lookback_years)


def assign_ranks(metrics: list[FundMetrics]) -> list[FundMetrics]:
    for metric in metrics:
        metric.target_std_dev = metric.annualized_return * TARGET_STD_DEV_TO_RETURN_RATIO
        metric.std_dev_gap = abs(metric.target_std_dev - metric.annualized_std_dev)
        metric.score = metric.annualized_return - metric.std_dev_gap

    ranked = sorted(
        metrics,
        key=lambda item: (
            -item.score,
            -item.annualized_return,
            item.std_dev_gap,
            item.annualized_std_dev,
            item.ticker,
        ),
    )
    for rank, metric in enumerate(ranked, start=1):
        metric.final_rank = rank
    return ranked


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def truncate_text(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    if max_length <= 3:
        return value[:max_length]
    return value[: max_length - 3].rstrip() + "..."


def format_table(
    metrics: Sequence[FundMetrics],
    lookback_years: int,
    *,
    universe_description: str,
    total_listings: int,
    total_ranked: int,
    skipped_count: int,
    displayed_count: int,
) -> str:
    rows = [
        [
            "Rank",
            "Ticker",
            "Name",
            "Score",
            "AvgAnnReturn",
            "AnnStdDev",
            "TargetStdDev",
            "StdDevGap",
            "Timeframe",
            "Years",
            "Points",
        ]
    ]
    for metric in metrics:
        rows.append(
            [
                str(metric.final_rank),
                metric.ticker,
                truncate_text(metric.name, 42),
                format_percent(metric.score),
                format_percent(metric.annualized_return),
                format_percent(metric.annualized_std_dev),
                format_percent(metric.target_std_dev),
                format_percent(metric.std_dev_gap),
                f"{metric.start_date.isoformat()} to {metric.end_date.isoformat()}",
                f"{metric.years_used:.2f}",
                str(metric.data_points),
            ]
        )

    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = [
        f"Fund universe: {universe_description}",
        "Price source: Yahoo Finance adjusted daily close",
        f"Lookback rule: use the most recent {lookback_years} years when available; otherwise use all available history",
        "Ranking rule: Score = AverageAnnReturn - abs((AverageAnnReturn * 0.45) - AnnStdDev); higher is better",
        f"Funds discovered: {total_listings}; ranked: {total_ranked}; skipped: {skipped_count}; displayed: {displayed_count}",
        "",
    ]
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)))
        if index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def metrics_to_dict(metric: FundMetrics) -> dict[str, object]:
    data = asdict(metric)
    data["start_date"] = metric.start_date.isoformat()
    data["end_date"] = metric.end_date.isoformat()
    return data


def write_csv(metrics: Iterable[FundMetrics], path: str) -> None:
    metrics = list(metrics)
    if not metrics:
        return

    fieldnames = list(metrics_to_dict(metrics[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metrics_to_dict(metric))


def resolve_fund_universe(args: argparse.Namespace) -> tuple[list[FundListing], str]:
    if args.sample and args.tickers:
        raise RuntimeError("Use either --sample or --tickers, not both")

    if args.sample:
        return (
            listings_from_tickers(SAMPLE_TICKERS),
            f"original sample tickers ({', '.join(SAMPLE_TICKERS)})",
        )

    if args.tickers:
        return listings_from_tickers(args.tickers), "custom ticker list"

    listings = fetch_fidelity_public_funds(
        args.fund_list_timeout,
        include_closed=args.include_closed,
        include_etfs=args.include_etfs,
    )
    fund_type = "mutual funds and ETFs" if args.include_etfs else "mutual funds"
    status = "all statuses" if args.include_closed else "open to new investors"
    as_of_dates = sorted(
        {listing.universe_as_of_date for listing in listings if listing.universe_as_of_date}
    )
    as_of = f"; Fidelity as-of date {', '.join(as_of_dates)}" if as_of_dates else ""
    return listings, f"Fidelity-managed retail {fund_type}, {status}{as_of}"


def collect_metrics(
    listings: Sequence[FundListing], args: argparse.Namespace
) -> tuple[list[FundMetrics], list[str]]:
    metrics: list[FundMetrics] = []
    errors: list[str] = []
    max_workers = min(args.workers, len(listings)) if listings else 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_by_listing = {
            executor.submit(
                calculate_listing_metrics,
                listing,
                args.lookback_years,
                args.timeout,
            ): listing
            for listing in listings
        }
        for future in concurrent.futures.as_completed(future_by_listing):
            listing = future_by_listing[future]
            try:
                metric = future.result()
                if metric.years_used < args.min_years:
                    errors.append(
                        f"{listing.ticker}: skipped; only {metric.years_used:.2f} years of usable history"
                    )
                else:
                    metrics.append(metric)
            except RuntimeError as exc:
                errors.append(str(exc))

    return metrics, errors


def print_error_summary(errors: Sequence[str], *, show_skipped: bool) -> None:
    if not errors:
        return

    if show_skipped or len(errors) <= 10:
        for error in errors:
            print(f"WARNING: {error}", file=sys.stderr)
        return

    for error in errors[:10]:
        print(f"WARNING: {error}", file=sys.stderr)
    print(
        f"WARNING: {len(errors) - 10} additional funds skipped; rerun with --show-skipped for details",
        file=sys.stderr,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.lookback_years <= 0:
        print("--lookback-years must be a positive integer", file=sys.stderr)
        return 2
    if args.timeout <= 0 or args.fund_list_timeout <= 0:
        print("--timeout and --fund-list-timeout must be positive numbers", file=sys.stderr)
        return 2
    if args.workers <= 0:
        print("--workers must be a positive integer", file=sys.stderr)
        return 2
    if args.min_years < 0:
        print("--min-years must be zero or greater", file=sys.stderr)
        return 2
    if args.top is not None and args.top <= 0:
        print("--top must be a positive integer", file=sys.stderr)
        return 2

    try:
        listings, universe_description = resolve_fund_universe(args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not listings:
        print("ERROR: no fund tickers found", file=sys.stderr)
        return 1

    metrics, errors = collect_metrics(listings, args)

    if not metrics:
        print_error_summary(errors, show_skipped=True)
        return 1

    ranked = assign_ranks(metrics)
    displayed = ranked[: args.top] if args.top is not None else ranked
    if args.json:
        print(json.dumps([metrics_to_dict(metric) for metric in ranked], indent=2))
    else:
        print(
            format_table(
                displayed,
                args.lookback_years,
                universe_description=universe_description,
                total_listings=len(listings),
                total_ranked=len(ranked),
                skipped_count=len(errors),
                displayed_count=len(displayed),
            )
        )
        print()
        print("Educational screening only; not investment advice.")

    if args.csv_path:
        write_csv(ranked, args.csv_path)

    print_error_summary(errors, show_skipped=args.show_skipped)
    return 1 if args.strict and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
