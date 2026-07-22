import pandas as pd

# Load the benchmark
df = pd.read_csv("data/financebench/benchmark.csv")
df["company"] = df["company"].str.upper()
# =============================================================================
# Basic statistics
# =============================================================================

print("=" * 80)
print("Dataset statistics")
print("=" * 80)

print(f"Total questions : {len(df)}")
print(f"Total companies : {df['company'].nunique()}")
print(f"Total documents : {df['source_document'].nunique()}")

# =============================================================================
# Questions per company
# =============================================================================

company_counts = (
    df["company"]
    .value_counts()
    .sort_values(ascending=False)
)

print("\nQuestions per company")
print(company_counts)

# =============================================================================
# Summary statistics
# =============================================================================

print("\nSummary")
print(company_counts.describe())

print(f"\nMinimum questions/company : {company_counts.min()}")
print(f"Maximum questions/company : {company_counts.max()}")

# =============================================================================
# Unique documents per company
# =============================================================================

docs_per_company = (
    df.groupby("company")["source_document"]
    .nunique()
    .sort_values(ascending=False)
)

print("\nUnique source documents per company")
print(docs_per_company)

# =============================================================================
# Detailed table
# =============================================================================

summary = (
    df.groupby("company")
      .agg(
          questions=("question_id", "count"),
          documents=("source_document", "nunique")
      )
      .sort_values("questions", ascending=False)
)

print("\nCompany summary")
print(summary)

# =============================================================================
# Check feasibility for different n_splits
# =============================================================================

print("\n" + "=" * 80)
print("Cross-validation feasibility")
print("=" * 80)

for n_splits in [3, 4, 5]:
    invalid = company_counts[company_counts < n_splits]

    print(f"\n{n_splits}-Fold CV")

    if len(invalid) == 0:
        print("✓ All companies satisfy the minimum requirement.")
    else:
        print("✗ The following companies have too few questions:")
        print(invalid)

# =============================================================================
# Optional: save statistics
# =============================================================================

summary.to_csv("company_distribution.csv")
print("\nSaved company_distribution.csv")