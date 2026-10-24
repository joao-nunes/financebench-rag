from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

# Short names and tickers that questions use instead of the company name.
ALIASES: dict[str, tuple[str, ...]] = {
    "American Express": ("Amex",),
    "Johnson & Johnson": ("JnJ", "J&J", "Johnson and Johnson"),
    "JPMorgan": ("JPM", "JP Morgan", "J.P. Morgan"),
    "MGM Resorts": ("MGM",),
    "Ulta Beauty": ("Ulta",),
    "PG&E Corporation": ("PG&E",),
    "AES Corporation": ("AES",),
    "Coca-Cola": ("Coke",),
    "Block": ("Square",),
    "McDonalds": ("McDonald",),
}

# APPLE_2022_10K, 3M_2023Q2_10Q, AMCOR_2022_8K_dated-2022-07-01,
# MCDONALDS_8K_dated-2023-01-06 (no period: fall back to the date).
_ID_PATTERN = re.compile(r"^(?P<prefix>.+?)_(?P<year>\d{4})(?:Q(?P<quarter>[1-4]))?_")
_DATED_PATTERN = re.compile(r"^(?P<prefix>.+?)_8K_dated[-_](?P<year>\d{4})-")

_YEAR = re.compile(r"(?<![\d.])(?:Q[1-4])?((?:19|20)\d{2})(?![\d.])")
_SHORT_YEAR = re.compile(r"\bFY\s?'?(\d{2})\b", re.IGNORECASE)
_QUARTER = re.compile(r"\bQ([1-4])|\b([1-4])Q\b", re.IGNORECASE)
_WORD_QUARTER = re.compile(
    r"\b(first|second|third|fourth)\s+(?:fiscal\s+)?quarter", re.IGNORECASE
)
_HALF = re.compile(r"\bH([12])\b", re.IGNORECASE)
_ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4}


@dataclass(frozen=True, slots=True)
class Filing:
    document_id: str
    company: str
    year: int | None
    quarter: int | None


@dataclass(frozen=True, slots=True)
class Period:
    year: int | None
    quarter: int | None


def parse_period(question: str) -> Period:
    """
    The latest fiscal year and the quarter a question refers to.

    Comparisons ("FY2021 vs FY2022") are answered from the latest filing,
    so the latest year mentioned is the one returned.
    """

    years = [int(y) for y in _YEAR.findall(question)]
    years += [2000 + int(y) for y in _SHORT_YEAR.findall(question)]

    quarter = None
    if match := _QUARTER.search(question):
        quarter = int(match.group(1) or match.group(2))
    elif match := _WORD_QUARTER.search(question):
        quarter = _ORDINALS[match.group(1).lower()]
    elif match := _HALF.search(question):
        quarter = 2 * int(match.group(1))

    return Period(year=max(years, default=None), quarter=quarter)


def parse_document_id(document_id: str) -> tuple[str, int | None, int | None]:
    """Return (prefix, year, quarter) encoded in a FinanceBench filing ID."""

    for pattern in (_ID_PATTERN, _DATED_PATTERN):
        if match := pattern.match(document_id):
            quarter = match.groupdict().get("quarter")
            return (
                match.group("prefix"),
                int(match.group("year")),
                int(quarter) if quarter else None,
            )

    return document_id.split("_")[0], None, None


def _normalize(text: str) -> str:
    # Drop possessives so "Adobe's" matches "Adobe".
    text = re.sub(r"['\u2019]s\b", "", text.lower())
    return re.sub(r"['\u2019]", "", text)


def _name_pattern(name: str) -> re.Pattern:
    # "Coca-Cola" also matches "Coca Cola".
    tokens = re.findall(r"[a-z0-9&]+", _normalize(name))
    body = r"[\s\-.]*".join(re.escape(t) for t in tokens)
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


class FilingFilter:
    """
    Detects which filings a question is about from its company and period.

    The company is matched by name or alias. Among that company's filings,
    those of the latest fiscal year mentioned are kept (or the latest
    earlier year, if there is no filing for it), narrowed to the quarter
    if one is mentioned. Without a year, all of the company's filings are
    kept. Returns None when no company is recognised, meaning: search
    everything.
    """

    def __init__(
        self,
        document_ids: Iterable[str],
        company_names: Mapping[str, str] | None = None,
    ) -> None:
        company_names = dict(company_names or {})

        # Name filings missing from company_names after another filing
        # with the same ID prefix (ADOBE_2022Q2_10Q -> ADOBE_2022_10K).
        by_prefix = {
            parse_document_id(doc_id)[0].upper(): name
            for doc_id, name in company_names.items()
        }

        self.filings: list[Filing] = []
        for doc_id in sorted(set(document_ids)):
            prefix, year, quarter = parse_document_id(doc_id)
            company = company_names.get(doc_id) or by_prefix.get(
                prefix.upper(), prefix.replace("_", " ").title()
            )
            self.filings.append(Filing(doc_id, company, year, quarter))

        self._patterns = {
            company: [_name_pattern(n) for n in (company, *ALIASES.get(company, ()))]
            for company in {f.company for f in self.filings}
        }

    @classmethod
    def from_document_information(
        cls,
        document_ids: Iterable[str],
        path: str | Path,
    ) -> FilingFilter:
        """Use the company names in financebench_document_information.jsonl."""

        with Path(path).open() as f:
            info = [json.loads(line) for line in f]

        return cls(document_ids, {d["doc_name"]: d["company"] for d in info})

    def companies(self, question: str) -> set[str]:
        text = _normalize(question)
        return {
            company
            for company, patterns in self._patterns.items()
            if any(p.search(text) for p in patterns)
        }

    def match(self, question: str) -> list[str] | None:
        companies = self.companies(question)
        if not companies:
            return None

        filings = [f for f in self.filings if f.company in companies]
        period = parse_period(question)

        if period.year is not None:
            years = {f.year for f in filings if f.year is not None}
            earlier = [y for y in years if y <= period.year]
            if earlier:
                year = max(earlier)
                filings = [f for f in filings if f.year == year]

        if period.quarter is not None:
            in_quarter = [f for f in filings if f.quarter == period.quarter]
            filings = in_quarter or filings

        return [f.document_id for f in filings]
