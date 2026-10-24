import pytest

from src.retrieval.filing_filter import FilingFilter, parse_period

FILINGS = [
    "3M_2021_10K",
    "3M_2022_10K",
    "3M_2023Q2_10Q",
    "BOEING_2022_10K",
    "JOHNSON_JOHNSON_2022_10K",
    "JOHNSON_JOHNSON_2023Q2_EARNINGS",
    "MCDONALDS_8K_dated-2023-01-06",
]
COMPANIES = {
    "3M_2022_10K": "3M",
    "BOEING_2022_10K": "Boeing",
    "JOHNSON_JOHNSON_2022_10K": "Johnson & Johnson",
    "MCDONALDS_8K_dated-2023-01-06": "McDonalds",
}


@pytest.fixture
def filing_filter():
    return FilingFilter(FILINGS, COMPANIES)


@pytest.mark.parametrize(
    ("question", "year", "quarter"),
    [
        ("How did revenue change from FY2021 to FY2022?", 2022, None),
        ("Did AMD report customer concentration in FY22?", 2022, None),
        ("What is the quick ratio for Q2 of FY2023?", 2023, 2),
        ("Revenue drop in Q22023 year over year?", 2023, 2),
        ("Investments in H1 FY2023?", 2023, 2),
        ("Is the business cyclical?", None, None),
    ],
)
def test_parse_period(question, year, quarter):
    period = parse_period(question)
    assert (period.year, period.quarter) == (year, quarter)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # Latest year mentioned; the possessive still matches the name.
        ("How did 3M's margin change from FY2021 to FY2022?", ["3M_2022_10K"]),
        # Quarter narrows to the 10-Q; filings missing from the company
        # names are named after filings with the same ID prefix.
        ("3M quick ratio for Q2 of FY2023?", ["3M_2023Q2_10Q"]),
        # No filing for FY2023: the latest earlier year.
        ("Boeing's production rate forecast for FY2023?", ["BOEING_2022_10K"]),
        # Alias, and an 8-K whose period comes from its date.
        ("Did JnJ's sales grow in Q2 of FY2023?", ["JOHNSON_JOHNSON_2023Q2_EARNINGS"]),
        ("What did McDonald's 8-K report in 2023?", ["MCDONALDS_8K_dated-2023-01-06"]),
        # No period: all of the company's filings.
        ("Is Boeing's business subject to cyclicality?", ["BOEING_2022_10K"]),
    ],
)
def test_match(filing_filter, question, expected):
    assert filing_filter.match(question) == expected


def test_match_without_company_searches_everything(filing_filter):
    assert filing_filter.match("Which nominees had the most votes against?") is None
