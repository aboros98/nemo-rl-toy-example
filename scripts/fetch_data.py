#!/usr/bin/env python3
"""Fetch and prepare CQL training data from three public sources.

SOURCE A — ByteRay-Labs/Query-Hub (MIT, YAML files with name/description/query)
SOURCE B — CrowdStrike/logscale-community-content (Queries-Only markdown)
SOURCE C — microsoft/NL2KQL (NLQ-KQL evaluation pairs, tagged as kql_transfer)

Cleans, deduplicates, splits 80/10/10, and saves as JSONL.
If < 100 training examples, generates synthetic pairs.
"""

import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

# Falcon event types with representative fields for schema context
FALCON_SCHEMAS: dict[str, list[str]] = {
    "ProcessRollup2": [
        "aid", "ComputerName", "ImageFileName", "CommandLine",
        "ParentBaseFileName", "ParentCommandLine", "SHA256HashData",
        "UserName", "UserSid", "ProcessStartTime", "ProcessEndTime",
    ],
    "NetworkConnectIP4": [
        "aid", "ComputerName", "RemoteAddressIP4", "RemotePort",
        "LocalAddressIP4", "LocalPort", "Protocol", "ConnectionDirection",
        "ImageFileName", "ContextProcessId",
    ],
    "DnsRequest": [
        "aid", "ComputerName", "DomainName", "RequestType",
        "RespondingDnsServer", "ImageFileName", "ContextProcessId",
    ],
    "UserLogon": [
        "aid", "ComputerName", "UserName", "LogonType",
        "LogonTime", "AuthenticationPackage", "RemoteAddressIP4",
        "UserSid", "LogonDomain",
    ],
    "UserAccountCreated": [
        "aid", "ComputerName", "UserName", "UserSid",
        "TargetUserName", "TargetUserSid", "CreatorProcessId",
    ],
    "SuspiciousProcessRollup2": [
        "aid", "ComputerName", "ImageFileName", "CommandLine",
        "ParentBaseFileName", "SuspiciousReason", "SHA256HashData",
        "UserName", "Severity",
    ],
    "PeFileWritten": [
        "aid", "ComputerName", "TargetFileName", "SHA256HashData",
        "WritingProcessImageFileName", "FileType", "Size",
    ],
    "ModuleLoadV3": [
        "aid", "ComputerName", "ImageFileName", "ModuleFileName",
        "SHA256HashData", "ContextProcessId",
    ],
}

# Security themes for synthetic generation
SECURITY_THEMES = ["explore", "detect", "hunt", "remediate", "report"]

# Synthetic CQL templates per theme and event type
SYNTHETIC_TEMPLATES: list[dict[str, str]] = [
    {
        "nl": "Find all {event_type} events for a specific hostname",
        "cql": '#event_simpleName={event_type} ComputerName="{hostname}" | head(100)',
        "theme": "explore",
    },
    {
        "nl": "Count {event_type} events grouped by ComputerName",
        "cql": "#event_simpleName={event_type} | groupBy(ComputerName, function=count())",
        "theme": "report",
    },
    {
        "nl": "Detect unusual {event_type} activity in the last 24 hours",
        "cql": '#event_simpleName={event_type} | timeChart(span=1h, function=count()) | sort(_count, order=desc)',
        "theme": "detect",
    },
    {
        "nl": "Hunt for {event_type} events with suspicious {field} values",
        "cql": '#event_simpleName={event_type} {field}="*suspicious*" | table({field}, ComputerName, aid)',
        "theme": "hunt",
    },
    {
        "nl": "Show top 10 {field} values from {event_type} events",
        "cql": "#event_simpleName={event_type} | top({field}, limit=10)",
        "theme": "explore",
    },
    {
        "nl": "List unique {field} values for {event_type} events on a given host",
        "cql": '#event_simpleName={event_type} ComputerName="{hostname}" | groupBy({field})',
        "theme": "explore",
    },
    {
        "nl": "Get average count of {event_type} events per hour over the last day",
        "cql": "#event_simpleName={event_type} | timeChart(span=1h, function=count()) | avg(_count)",
        "theme": "report",
    },
    {
        "nl": "Find {event_type} events where {field} contains a specific pattern",
        "cql": '#event_simpleName={event_type} | regex("{field}", regex=".*pattern.*") | head(50)',
        "theme": "hunt",
    },
    {
        "nl": "Count distinct {field} values in {event_type} events",
        "cql": '#event_simpleName={event_type} | count({field}, distinct=true)',
        "theme": "report",
    },
    {
        "nl": "Show the timeline of {event_type} events for a specific user",
        "cql": '#event_simpleName={event_type} UserName="{user}" | sort(ProcessStartTime, order=asc) | table(ProcessStartTime, {field}, ComputerName)',
        "theme": "hunt",
    },
    {
        "nl": "Identify top hosts generating {event_type} events",
        "cql": "#event_simpleName={event_type} | groupBy(ComputerName, function=count()) | sort(_count, order=desc) | head(20)",
        "theme": "detect",
    },
    {
        "nl": "Summarize {event_type} event volume by day",
        "cql": "#event_simpleName={event_type} | timeChart(span=1d, function=count())",
        "theme": "report",
    },
]

SAMPLE_HOSTNAMES = [
    "WORKSTATION-01", "DC-PRIMARY", "SERVER-WEB01", "LAPTOP-DEV42",
    "HOST-FINANCE03", "ENDPOINT-HR07",
]

SAMPLE_USERS = [
    "admin", "jdoe", "svc_account", "analyst01", "root",
]


def clone_repo(url: str, dest: Path) -> bool:
    """Clone a git repo if not already present."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already cloned")
        return True
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"  [OK] Cloned {url}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [WARN] Failed to clone {url}: {e.stderr.strip()}")
        return False


def extract_schema_context(cql: str) -> str:
    """Extract schema context from a CQL query — event types and field names."""
    context_parts = []

    # Find event type references
    event_types = re.findall(r"#?event_simpleName\s*=\s*(\w+)", cql)
    event_types += re.findall(r"#?type\s*=\s*(\w+)", cql)
    for et in event_types:
        if et in FALCON_SCHEMAS:
            fields = FALCON_SCHEMAS[et]
            context_parts.append(f"{et}: {', '.join(fields[:6])}")

    # Find field references used in the query
    field_refs = set(re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:[A-Z][a-z]+)+)\b", cql))
    if field_refs:
        context_parts.append(f"fields: {', '.join(sorted(field_refs)[:8])}")

    return "; ".join(context_parts) if context_parts else "general CQL query"


def fetch_source_a(repos_dir: Path) -> list[dict]:
    """Fetch from ByteRay-Labs/Query-Hub."""
    print("\n--- SOURCE A: ByteRay-Labs/Query-Hub ---")
    dest = repos_dir / "Query-Hub"
    if not clone_repo("https://github.com/ByteRay-Labs/Query-Hub.git", dest):
        return []

    records = []
    yaml_files = list(dest.rglob("*.yaml")) + list(dest.rglob("*.yml"))
    print(f"  Found {len(yaml_files)} YAML files")

    try:
        import yaml
    except ImportError:
        print("  [WARN] PyYAML not installed, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyyaml"],
                       capture_output=True)
        import yaml

    for yf in yaml_files:
        try:
            with open(yf) as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            name = data.get("name", "")
            desc = data.get("description", "")
            query = data.get("query", "") or data.get("cql", "")
            if not query or not (name or desc):
                continue
            nl = f"{name}: {desc}" if name and desc else (name or desc)
            records.append({
                "nl_query": nl.strip(),
                "cql_query": query.strip(),
                "source": "query_hub",
                "tags": ["logscale", "community"],
                "schema_context": extract_schema_context(query),
            })
        except Exception:
            continue

    print(f"  Extracted {len(records)} records")
    return records


def fetch_source_b(repos_dir: Path) -> list[dict]:
    """Fetch from CrowdStrike/logscale-community-content (Queries-Only)."""
    print("\n--- SOURCE B: CrowdStrike/logscale-community-content ---")
    dest = repos_dir / "logscale-community-content"
    if not clone_repo(
        "https://github.com/CrowdStrike/logscale-community-content.git", dest
    ):
        return []

    records = []
    # Look for Queries-Only directory or similar
    queries_dirs = list(dest.rglob("Queries-Only")) + list(dest.rglob("queries"))
    md_files = []
    for qd in queries_dirs:
        md_files.extend(qd.rglob("*.md"))

    # Also check top-level for any markdown with queries
    if not md_files:
        md_files = list(dest.rglob("*.md"))

    print(f"  Found {len(md_files)} markdown files")

    for mf in md_files:
        try:
            content = mf.read_text(errors="ignore")
            records.extend(_parse_markdown_queries(content))
        except Exception:
            continue

    print(f"  Extracted {len(records)} records")
    return records


def _parse_markdown_queries(content: str) -> list[dict]:
    """Parse markdown to extract heading + CQL code block pairs."""
    records = []
    lines = content.split("\n")
    current_heading = ""
    in_code_block = False
    code_block_lines = []

    for line in lines:
        # Detect headings
        if line.startswith("#"):
            heading_text = line.lstrip("#").strip()
            if heading_text:
                current_heading = heading_text

        # Detect code blocks
        if line.strip().startswith("```"):
            if in_code_block:
                # End of code block
                query = "\n".join(code_block_lines).strip()
                if query and "|" in query and current_heading:
                    records.append({
                        "nl_query": current_heading,
                        "cql_query": query,
                        "source": "logscale_community",
                        "tags": ["crowdstrike", "community"],
                        "schema_context": extract_schema_context(query),
                    })
                code_block_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_block_lines = []
        elif in_code_block:
            code_block_lines.append(line)

    return records


def fetch_source_c(repos_dir: Path) -> list[dict]:
    """Fetch from microsoft/NL2KQL (KQL transfer learning pairs)."""
    print("\n--- SOURCE C: microsoft/NL2KQL ---")
    dest = repos_dir / "NL2KQL"
    if not clone_repo("https://github.com/microsoft/NL2KQL.git", dest):
        return []

    records = []

    # Look for evaluation data files
    data_files = (
        list(dest.rglob("*.jsonl"))
        + list(dest.rglob("*.json"))
        + list(dest.rglob("*.csv"))
        + list(dest.rglob("*.tsv"))
    )
    print(f"  Found {len(data_files)} data files")

    for df in data_files:
        try:
            if df.suffix == ".jsonl":
                records.extend(_parse_jsonl_kql(df))
            elif df.suffix == ".json":
                records.extend(_parse_json_kql(df))
            elif df.suffix in (".csv", ".tsv"):
                records.extend(_parse_csv_kql(df))
        except Exception:
            continue

    print(f"  Extracted {len(records)} records")
    return records


def _parse_jsonl_kql(path: Path) -> list[dict]:
    """Parse JSONL files for NL-KQL pairs."""
    records = []
    with open(path) as f:
        for line in f:
            try:
                data = json.loads(line)
                nl = (
                    data.get("nl", "")
                    or data.get("question", "")
                    or data.get("NL", "")
                    or data.get("context", "")  # NL2KQL uses 'context'
                )
                kql = (
                    data.get("kql", "")
                    or data.get("query", "")
                    or data.get("KQL", "")
                    or data.get("baseline", "")  # NL2KQL uses 'baseline'
                )
                if nl and kql:
                    records.append({
                        "nl_query": nl.strip(),
                        "cql_query": kql.strip(),
                        "source": "kql_transfer",
                        "tags": ["kql", "transfer_learning"],
                        "schema_context": extract_schema_context(kql),
                    })
            except json.JSONDecodeError:
                continue
    return records


def _parse_json_kql(path: Path) -> list[dict]:
    """Parse JSON files for NL-KQL pairs."""
    records = []
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    nl = item.get("nl", item.get("question", item.get("NL", "")))
                    kql = item.get("kql", item.get("query", item.get("KQL", "")))
                    if nl and kql:
                        records.append({
                            "nl_query": nl.strip(),
                            "cql_query": kql.strip(),
                            "source": "kql_transfer",
                            "tags": ["kql", "transfer_learning"],
                            "schema_context": extract_schema_context(kql),
                        })
    except Exception:
        pass
    return records


def _parse_csv_kql(path: Path) -> list[dict]:
    """Parse CSV/TSV files for NL-KQL pairs."""
    import csv

    records = []
    delimiter = "\t" if path.suffix == ".tsv" else ","
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                # Try common column name variants
                nl = (
                    row.get("nl", "")
                    or row.get("NL", "")
                    or row.get("question", "")
                    or row.get("Question", "")
                    or row.get("natural_language", "")
                )
                kql = (
                    row.get("kql", "")
                    or row.get("KQL", "")
                    or row.get("query", "")
                    or row.get("Query", "")
                )
                if nl and kql:
                    records.append({
                        "nl_query": nl.strip(),
                        "cql_query": kql.strip(),
                        "source": "kql_transfer",
                        "tags": ["kql", "transfer_learning"],
                        "schema_context": extract_schema_context(kql),
                    })
    except Exception:
        pass
    return records


def deduplicate(records: list[dict]) -> list[dict]:
    """Deduplicate records by normalized CQL query hash."""
    seen = set()
    unique = []
    for rec in records:
        # Normalize: lowercase, collapse whitespace
        norm = re.sub(r"\s+", " ", rec["cql_query"].strip().lower())
        h = hashlib.md5(norm.encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(rec)
    return unique


def generate_synthetic_pairs(target_count: int, existing: list[dict]) -> list[dict]:
    """Generate synthetic NL-CQL pairs using templates.

    Uses template-based generation (no LLM required for the dummy pipeline).
    In production, use LLM generation + back-translation + Jaccard ≥ 0.7
    validation per the NL2KQL paper method.
    """
    print(f"\n--- Generating synthetic pairs (target: {target_count}) ---")
    print("  [MOCK] Using template-based generation (no LLM call)")
    print("  [MOCK] Skipping Jaccard ≥ 0.7 back-translation validation")
    print("         (requires LLM; templates are hand-crafted to be aligned)")

    synthetic = []
    attempts = 0
    max_attempts = target_count * 10

    event_types = list(FALCON_SCHEMAS.keys())
    existing_hashes = {
        hashlib.md5(
            re.sub(r"\s+", " ", r["cql_query"].strip().lower()).encode()
        ).hexdigest()
        for r in existing
    }

    while len(synthetic) < target_count and attempts < max_attempts:
        attempts += 1

        template = random.choice(SYNTHETIC_TEMPLATES)
        event_type = random.choice(event_types)
        fields = FALCON_SCHEMAS[event_type]
        field = random.choice(fields[1:])  # Skip 'aid'
        hostname = random.choice(SAMPLE_HOSTNAMES)
        user = random.choice(SAMPLE_USERS)

        nl = template["nl"].format(
            event_type=event_type,
            field=field,
            hostname=hostname,
            user=user,
        )
        cql = template["cql"].format(
            event_type=event_type,
            field=field,
            hostname=hostname,
            user=user,
        )

        # Template-based pairs skip Jaccard back-translation validation
        # (which requires LLM). In production with LLM generation:
        #   back_nl = llm_back_translate(cql)
        #   jaccard = jaccard_similarity(nl, back_nl)
        #   if jaccard < 0.7: continue

        # Deduplicate against existing
        norm = re.sub(r"\s+", " ", cql.strip().lower())
        h = hashlib.md5(norm.encode()).hexdigest()
        if h not in existing_hashes:
            existing_hashes.add(h)
            synthetic.append({
                "nl_query": nl,
                "cql_query": cql,
                "source": "synthetic",
                "tags": ["synthetic", template["theme"]],
                "schema_context": f"{event_type}: {', '.join(fields[:6])}",
            })

    print(f"  Generated {len(synthetic)} synthetic pairs")
    return synthetic


def split_data(
    records: list[dict],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split records into train/val/test with given ratios."""
    random.seed(seed)
    shuffled = list(records)
    random.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Save records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"  Saved {len(records)} records to {path}")


def print_stats(train: list, val: list, test: list) -> None:
    """Print dataset statistics."""
    all_records = train + val + test
    sources = {}
    for rec in all_records:
        src = rec["source"]
        sources[src] = sources.get(src, 0) + 1

    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  Total records:   {len(all_records)}")
    print(f"  Train:           {len(train)}")
    print(f"  Validation:      {len(val)}")
    print(f"  Test:            {len(test)}")
    print(f"\n  By source:")
    for src, count in sorted(sources.items()):
        print(f"    {src:25s} {count:5d}")
    print("=" * 60)


def _is_natural_language(text: str) -> bool:
    """Return True if text looks like natural language, not code."""
    text = text.strip()
    if not text:
        return False
    # Must have at least 3 space-separated words
    if len(text.split()) < 3:
        return False
    # Reject if it starts with typical code patterns
    if re.match(r"^[#@]?\w+\s*[=/<>!]", text):
        return False
    # Reject if more than half the tokens contain operators
    tokens = text.split()
    code_tokens = sum(1 for t in tokens if re.search(r"[=|(){}\[\]<>]", t))
    if code_tokens > len(tokens) * 0.5:
        return False
    return True


def _format_user_prompt(schema_context: str, nl_query: str) -> str:
    """Format the user message for NeMo RL. System prompt is separate."""
    parts = []
    if schema_context and schema_context != "general CQL query":
        parts.append(f"Schema: {schema_context}")
    parts.append(f"Request: {nl_query}")
    return "\n".join(parts)


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    repos_dir = data_dir / "repos"
    repos_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching CQL training data from public sources...")

    # Fetch from all three sources
    records_a = fetch_source_a(repos_dir)
    records_b = fetch_source_b(repos_dir)
    records_c = fetch_source_c(repos_dir)

    all_records = records_a + records_b + records_c
    print(f"\nTotal raw records: {len(all_records)}")

    # Clean and deduplicate
    all_records = [r for r in all_records if r["cql_query"].strip()]

    # Filter out entries where nl_query is code, not natural language.
    # The logscale_community source extracts markdown headings that are
    # often CQL fragments (e.g. "event_simpleName=DetectionExcluded").
    # These poison RL training by teaching the model to copy input→output.
    before_nl_filter = len(all_records)
    all_records = [r for r in all_records if _is_natural_language(r["nl_query"])]
    filtered = before_nl_filter - len(all_records)
    if filtered:
        print(f"  Filtered {filtered} records with code-as-NL (not natural language)")

    all_records = deduplicate(all_records)
    print(f"After deduplication: {len(all_records)}")

    # Format nl_query to include schema context (model needs it for correct field names).
    # System prompt is a separate file; nl_query is the full user message.
    for rec in all_records:
        schema = rec.get("schema_context", "")
        if schema and schema != "general CQL query":
            rec["nl_query"] = f"Schema: {schema}\n{rec['nl_query']}"

    # Split
    train, val, test = split_data(all_records)

    # If training set is too small, generate synthetic pairs
    if len(train) < 100:
        needed = 100 - len(train)
        synthetic = generate_synthetic_pairs(
            max(needed, 50), all_records
        )
        train.extend(synthetic)
        # Re-split to maintain ratios with the new data
        all_records = train + val + test
        train, val, test = split_data(all_records)

    # Ensure we have at least some validation and test data
    if len(val) < 5:
        synthetic_val = generate_synthetic_pairs(10, all_records)
        val.extend(synthetic_val)
    if len(test) < 5:
        synthetic_test = generate_synthetic_pairs(10, all_records + val)
        test.extend(synthetic_test)

    # Save
    save_jsonl(train, data_dir / "train.jsonl")
    save_jsonl(val, data_dir / "val.jsonl")
    save_jsonl(test, data_dir / "test.jsonl")

    print_stats(train, val, test)
    print("\nDone!")


if __name__ == "__main__":
    main()
