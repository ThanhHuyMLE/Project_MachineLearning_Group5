#!/usr/bin/env python3
"""Automated commit message and PR title validator for CI and local workflows.

Enforces the commit message convention defined in docs/commit-convention.md:
    TYPE(scope): summary
or
    TYPE(scope)!: summary
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from typing import NamedTuple

ALLOWED_TYPES: set[str] = {
    "FEAT",
    "FIX",
    "REFACT",
    "DATA",
    "PERF",
    "TEST",
    "DOCS",
    "STYLE",
    "CHORE",
    "SEC",
    "OPS",
    "REMOVE",
}

TYPE_NORMALIZATION: dict[str, str] = {
    "REFACTOR": "REFACT",
    "REFACTORING": "REFACT",
    "DOC": "DOCS",
    "TESTS": "TEST",
    "PERFORMANCE": "PERF",
    "SECURITY": "SEC",
    "OPERATIONS": "OPS",
    "OPERATION": "OPS",
    "DELETION": "REMOVE",
    "DELETE": "REMOVE",
}

SUBJECT_REGEX = re.compile(
    r"^(?P<type>[A-Za-z0-9_-]+)"
    r"(?:\((?P<scope>[^)]*)\))?"
    r"(?P<breaking>!)?"
    r"(?P<colon>:)"
    r"(?P<space>\s*)"
    r"(?P<summary>.*)$"
)

PROHIBITED_VAGUE_SUMMARIES: set[str] = {
    "fix",
    "fixes",
    "fix bug",
    "fix bugs",
    "bug fix",
    "bugfix",
    "bug fixes",
    "fixes bug",
    "update",
    "updates",
    "update code",
    "updating",
    "updating code",
    "change",
    "changes",
    "changing",
    "refactor",
    "refactoring",
    "improve",
    "improves",
    "improve stuff",
    "improvements",
    "cleanup",
    "cleaning",
    "clean up",
    "wip",
    "misc",
    "minor changes",
    "quick fix",
    "more changes",
    "various fixes",
    "various changes",
    "stuff",
}

PAST_TENSE_VERBS: dict[str, str] = {
    "added": "add",
    "adjusted": "adjust",
    "allowed": "allow",
    "amended": "amend",
    "analyzed": "analyze",
    "applied": "apply",
    "automated": "automate",
    "avoided": "avoid",
    "blocked": "block",
    "built": "build",
    "bumped": "bump",
    "calculated": "calculate",
    "called": "call",
    "changed": "change",
    "checked": "check",
    "clarified": "clarify",
    "cleaned": "clean",
    "cleared": "clear",
    "closed": "close",
    "combined": "combine",
    "configured": "configure",
    "converted": "convert",
    "corrected": "correct",
    "created": "create",
    "debugged": "debug",
    "decreased": "decrease",
    "defined": "define",
    "deleted": "delete",
    "deprecating": "deprecate",
    "deprecated": "deprecate",
    "disabled": "disable",
    "documented": "document",
    "dropped": "drop",
    "enabled": "enable",
    "enforced": "enforce",
    "enhanced": "enhance",
    "ensured": "ensure",
    "evaluated": "evaluate",
    "executed": "execute",
    "expanded": "expand",
    "exported": "export",
    "extracted": "extract",
    "factored": "factor",
    "fixed": "fix",
    "formatted": "format",
    "handled": "handle",
    "ignored": "ignore",
    "implemented": "implement",
    "improved": "improve",
    "included": "include",
    "increased": "increase",
    "initialized": "initialize",
    "installed": "install",
    "integrated": "integrate",
    "introduced": "introduce",
    "isolated": "isolate",
    "loaded": "load",
    "logged": "log",
    "maintained": "maintain",
    "managed": "manage",
    "merged": "merge",
    "migrated": "migrate",
    "minified": "minify",
    "modified": "modify",
    "moved": "move",
    "normalized": "normalize",
    "optimized": "optimize",
    "organized": "organize",
    "parsed": "parse",
    "patched": "patch",
    "persisted": "persist",
    "pinned": "pin",
    "polished": "polish",
    "prepared": "prepare",
    "prevented": "prevent",
    "printed": "print",
    "processed": "process",
    "pruned": "prune",
    "published": "publish",
    "pulled": "pull",
    "pushed": "push",
    "raised": "raise",
    "read": "read",
    "rebased": "rebase",
    "rebuilt": "rebuild",
    "refactored": "refactor",
    "referenced": "reference",
    "reformatted": "reformat",
    "released": "release",
    "reloaded": "reload",
    "removed": "remove",
    "renamed": "rename",
    "reorganized": "reorganize",
    "replaced": "replace",
    "reported": "report",
    "reset": "reset",
    "resolved": "resolve",
    "restored": "restore",
    "restricted": "restrict",
    "restructured": "restructure",
    "retained": "retain",
    "returned": "return",
    "reverted": "revert",
    "reviewed": "review",
    "revised": "revise",
    "rewrote": "rewrite",
    "ran": "run",
    "saved": "save",
    "separated": "separate",
    "simplified": "simplify",
    "skipped": "skip",
    "standardized": "standardize",
    "started": "start",
    "stopped": "stop",
    "stripped": "strip",
    "supported": "support",
    "switched": "switch",
    "synchronized": "synchronize",
    "synced": "sync",
    "tested": "test",
    "tracked": "track",
    "trimmed": "trim",
    "tuned": "tune",
    "unified": "unify",
    "updated": "update",
    "upgraded": "upgrade",
    "used": "use",
    "validated": "validate",
    "verified": "verify",
    "wrote": "write",
}

GERUND_VERBS: dict[str, str] = {
    "adding": "add",
    "adjusting": "adjust",
    "allowing": "allow",
    "amending": "amend",
    "analyzing": "analyze",
    "applying": "apply",
    "automating": "automate",
    "avoiding": "avoid",
    "blocking": "block",
    "building": "build",
    "bumping": "bump",
    "calculating": "calculate",
    "calling": "call",
    "changing": "change",
    "checking": "check",
    "clarifying": "clarify",
    "cleaning": "clean",
    "clearing": "clear",
    "closing": "close",
    "combining": "combine",
    "configuring": "configure",
    "converting": "convert",
    "correcting": "correct",
    "creating": "create",
    "debugging": "debug",
    "decreasing": "decrease",
    "defining": "define",
    "deleting": "delete",
    "disabling": "disable",
    "documenting": "document",
    "dropping": "drop",
    "enabling": "enable",
    "enforcing": "enforce",
    "enhancing": "enhance",
    "ensuring": "ensure",
    "evaluating": "evaluate",
    "executing": "execute",
    "expanding": "expand",
    "exporting": "export",
    "extracting": "extract",
    "factoring": "factor",
    "fixing": "fix",
    "formatting": "format",
    "handling": "handle",
    "ignoring": "ignore",
    "implementing": "implement",
    "improving": "improve",
    "including": "include",
    "increasing": "increase",
    "initializing": "initialize",
    "installing": "install",
    "integrating": "integrate",
    "introducing": "introduce",
    "isolating": "isolate",
    "loading": "load",
    "logging": "log",
    "maintaining": "maintain",
    "managing": "manage",
    "merging": "merge",
    "migrating": "migrate",
    "minifying": "minify",
    "modifying": "modify",
    "moving": "move",
    "normalizing": "normalize",
    "optimizing": "optimize",
    "organizing": "organize",
    "parsing": "parse",
    "patching": "patch",
    "persisting": "persist",
    "pinning": "pin",
    "polishing": "polish",
    "preparing": "prepare",
    "preventing": "prevent",
    "printing": "print",
    "processing": "process",
    "pruning": "prune",
    "publishing": "publish",
    "pulling": "pull",
    "pushing": "push",
    "raising": "raise",
    "reading": "read",
    "rebasing": "rebase",
    "rebuilding": "rebuild",
    "refactoring": "refactor",
    "referencing": "reference",
    "reformatting": "reformat",
    "releasing": "release",
    "reloading": "reload",
    "removing": "remove",
    "renaming": "rename",
    "reorganizing": "reorganize",
    "replacing": "replace",
    "reporting": "report",
    "resetting": "reset",
    "resolving": "resolve",
    "restoring": "restore",
    "restricting": "restrict",
    "restructuring": "restructure",
    "retaining": "retain",
    "returning": "return",
    "reverting": "revert",
    "reviewing": "review",
    "revising": "revise",
    "rewriting": "rewrite",
    "running": "run",
    "saving": "save",
    "separating": "separate",
    "simplifying": "simplify",
    "skipping": "skip",
    "standardizing": "standardize",
    "starting": "start",
    "stopping": "stop",
    "stripping": "strip",
    "supporting": "support",
    "switching": "switch",
    "synchronizing": "synchronize",
    "syncing": "sync",
    "testing": "test",
    "tracking": "track",
    "trimming": "trim",
    "tuning": "tune",
    "unifying": "unify",
    "updating": "update",
    "upgrading": "upgrade",
    "using": "use",
    "validating": "validate",
    "verifying": "verify",
    "writing": "write",
}

THIRD_PERSON_VERBS: dict[str, str] = {
    "adds": "add",
    "adjusts": "adjust",
    "allows": "allow",
    "amends": "amend",
    "analyzes": "analyze",
    "applies": "apply",
    "automates": "automate",
    "avoids": "avoid",
    "blocks": "block",
    "builds": "build",
    "bumps": "bump",
    "calculates": "calculate",
    "calls": "call",
    "changes": "change",
    "checks": "check",
    "clarifies": "clarify",
    "cleans": "clean",
    "clears": "clear",
    "closes": "close",
    "combines": "combine",
    "configures": "configure",
    "converts": "convert",
    "corrects": "correct",
    "creates": "create",
    "debugs": "debug",
    "decreases": "decrease",
    "defines": "define",
    "deletes": "delete",
    "deprecates": "deprecate",
    "disables": "disable",
    "documents": "document",
    "drops": "drop",
    "enables": "enable",
    "enforces": "enforce",
    "enhances": "enhance",
    "ensures": "ensure",
    "evaluates": "evaluate",
    "executes": "execute",
    "expands": "expand",
    "exports": "export",
    "extracts": "extract",
    "factors": "factor",
    "fixes": "fix",
    "formats": "format",
    "handles": "handle",
    "ignores": "ignore",
    "implements": "implement",
    "improves": "improve",
    "includes": "include",
    "increases": "increase",
    "initializes": "initialize",
    "installs": "install",
    "integrates": "integrate",
    "introduces": "introduce",
    "isolates": "isolate",
    "loads": "load",
    "logs": "log",
    "maintains": "maintain",
    "manages": "manage",
    "merges": "merge",
    "migrates": "migrate",
    "minifies": "minify",
    "modifies": "modify",
    "moves": "move",
    "normalizes": "normalize",
    "optimizes": "optimize",
    "organizes": "organize",
    "parses": "parse",
    "patches": "patch",
    "persists": "persist",
    "pins": "pin",
    "polishes": "polish",
    "prepares": "prepare",
    "prevents": "prevent",
    "prints": "print",
    "processes": "process",
    "prunes": "prune",
    "publishes": "publish",
    "pulls": "pull",
    "pushes": "push",
    "raises": "raise",
    "reads": "read",
    "rebases": "rebase",
    "rebuilds": "rebuild",
    "refactors": "refactor",
    "references": "reference",
    "reformats": "reformat",
    "releases": "release",
    "reloads": "reload",
    "removes": "remove",
    "renames": "rename",
    "reorganizes": "reorganize",
    "replaces": "replace",
    "reports": "report",
    "resets": "reset",
    "resolves": "resolve",
    "restores": "restore",
    "restricts": "restrict",
    "restructures": "restructure",
    "retains": "retain",
    "returns": "return",
    "reverts": "revert",
    "reviews": "review",
    "revises": "revise",
    "rewrites": "rewrite",
    "runs": "run",
    "saves": "save",
    "separates": "separate",
    "simplifies": "simplify",
    "skips": "skip",
    "standardizes": "standardize",
    "starts": "start",
    "stops": "stop",
    "strips": "strip",
    "supports": "support",
    "switches": "switch",
    "synchronizes": "synchronize",
    "syncs": "sync",
    "tests": "test",
    "tracks": "track",
    "trims": "trim",
    "tunes": "tune",
    "unifies": "unify",
    "updates": "update",
    "upgrades": "upgrade",
    "uses": "use",
    "validates": "validate",
    "verifies": "verify",
    "writes": "write",
}


class ValidationResult(NamedTuple):
    sha: str
    message: str
    errors: list[str]
    suggested: str


def validate_commit_message(message: str, is_pr_title: bool = False) -> list[str]:
    """Validate a commit message or PR title against the project convention.

    Returns an empty list if valid, or a list of specific rule violation descriptions.
    """
    if not message or not message.strip():
        target = "PR title" if is_pr_title else "Commit message"
        return [f"{target} cannot be empty."]

    lines = message.strip().splitlines()
    subject = lines[0].strip()

    # If validating a PR title, strip any trailing GitHub PR number suffix e.g. ' (#123)'
    if is_pr_title:
        subject = re.sub(r"\s*\(\s*#\d+\s*\)$", "", subject).strip()

    errors: list[str] = []

    match = SUBJECT_REGEX.match(subject)
    if not match:
        if ":" not in subject:
            errors.append(
                "Missing colon separator after TYPE(scope). Expected 'TYPE(scope): summary'."
            )
        elif "(" not in subject or ")" not in subject:
            errors.append("Invalid scope syntax. Expected 'TYPE(scope): summary'.")
        else:
            errors.append(
                "Invalid format. Expected 'TYPE(scope): summary' or 'TYPE(scope)!: summary'."
            )
        return errors

    type_part = match.group("type")
    scope_part = match.group("scope")
    space_part = match.group("space")
    summary_part = match.group("summary")

    # 1. TYPE checks
    if type_part in ALLOWED_TYPES:
        pass
    elif type_part.upper() in ALLOWED_TYPES:
        errors.append(
            f"TYPE must be uppercase (found '{type_part}', expected '{type_part.upper()}')."
        )
    elif type_part.upper() in TYPE_NORMALIZATION:
        normalized = TYPE_NORMALIZATION[type_part.upper()]
        errors.append(f"TYPE must be uppercase '{normalized}' (found '{type_part}').")
    else:
        allowed_list = ", ".join(sorted(ALLOWED_TYPES))
        errors.append(f"Unsupported TYPE '{type_part}'. Allowed types: {allowed_list}.")

    # 2. Scope checks
    if scope_part is None:
        errors.append("Scope is required: TYPE(scope): summary.")
    else:
        scope = scope_part.strip()
        if not scope:
            errors.append("Scope cannot be empty.")
        else:
            if any(c.isupper() for c in scope):
                errors.append(f"Scope must be lowercase (found '{scope}').")
            elif not re.match(r"^[a-z0-9_/-]+$", scope):
                errors.append(
                    f"Scope contains invalid characters (found '{scope}'). "
                    "Use lowercase alphanumeric characters, hyphens, underscores, or slashes."
                )

    # 3. Colon and space checks
    if not space_part:
        errors.append("A space is required after ':' (e.g., 'TYPE(scope): summary').")

    # 4. Summary checks
    summary = summary_part.strip()
    if not summary:
        errors.append("Summary is required and cannot be empty.")
    else:
        if summary.endswith("."):
            errors.append("Summary must not end with a period ('.').")

        # Prohibited vague summaries check
        clean_summary = summary.lower().rstrip(".,!?;: ")
        if clean_summary in PROHIBITED_VAGUE_SUMMARIES or any(
            clean_summary.startswith(f"{vague} ")
            for vague in ("fix bug", "update code", "improve stuff", "various fixes")
        ):
            errors.append(
                f"Summary '{summary}' is too vague. Provide a specific, action-oriented description."
            )

        # Imperative mood checks
        words = re.findall(r"[A-Za-z]+", summary)
        if words:
            first_word = words[0].lower()
            if first_word in PAST_TENSE_VERBS:
                suggested_verb = PAST_TENSE_VERBS[first_word]
                errors.append(
                    f"Summary should use imperative mood (found past tense '{words[0]}', consider '{suggested_verb}')."
                )
            elif first_word in GERUND_VERBS:
                suggested_verb = GERUND_VERBS[first_word]
                errors.append(
                    f"Summary should use imperative mood (found gerund '{words[0]}', consider '{suggested_verb}')."
                )
            elif first_word in THIRD_PERSON_VERBS:
                suggested_verb = THIRD_PERSON_VERBS[first_word]
                errors.append(
                    f"Summary should use imperative mood (found 3rd person '{words[0]}', consider '{suggested_verb}')."
                )
            elif (
                first_word.endswith("ed")
                and len(first_word) > 4
                and first_word not in {"speed", "embed", "feed", "bleed", "breed", "seed"}
            ):
                errors.append(
                    f"Summary should use imperative mood (found possible past tense '{words[0]}')."
                )
            elif (
                first_word.endswith("ing")
                and len(first_word) > 5
                and first_word not in {"string", "spring", "ring", "bring", "wing"}
            ):
                errors.append(
                    f"Summary should use imperative mood (found possible gerund '{words[0]}')."
                )

    # 5. Multiline message check (blank line between subject and body)
    if not is_pr_title and len(lines) > 1 and lines[1].strip() != "":
        errors.append("Separate the subject from the commit body with a blank line.")

    return errors


def suggest_valid_commit(message: str, is_pr_title: bool = False) -> str:
    """Generate a compliant commit message suggestion based on an invalid one."""
    if not message or not message.strip():
        return "FEAT(model): add initial classifier pipeline"

    lines = message.strip().splitlines()
    subject = lines[0].strip()
    if is_pr_title:
        subject = re.sub(r"\s*\(\s*#\d+\s*\)$", "", subject).strip()

    match = SUBJECT_REGEX.match(subject)
    if not match:
        return "FEAT(model): add descriptive summary"

    type_part = match.group("type")
    scope_part = match.group("scope")
    breaking_part = match.group("breaking") or ""
    summary_part = match.group("summary").strip()

    # Normalize type
    suggested_type = type_part.upper()
    suggested_type = TYPE_NORMALIZATION.get(suggested_type, suggested_type)
    if suggested_type not in ALLOWED_TYPES:
        suggested_type = "FEAT"

    # Normalize scope
    if scope_part is None or not scope_part.strip():
        suggested_scope = "model"
    else:
        suggested_scope = scope_part.strip().lower()
        suggested_scope = re.sub(r"[^a-z0-9_/-]", "-", suggested_scope).strip("-") or "model"

    # Normalize summary
    suggested_summary = summary_part.rstrip(". ")
    if not suggested_summary or suggested_summary.lower() in PROHIBITED_VAGUE_SUMMARIES:
        suggested_summary = "add specific descriptive action"
    else:
        words = re.findall(r"[A-Za-z]+", suggested_summary)
        if words:
            fw = words[0].lower()
            replacement = (
                PAST_TENSE_VERBS.get(fw) or GERUND_VERBS.get(fw) or THIRD_PERSON_VERBS.get(fw)
            )
            if replacement:
                suggested_summary = re.sub(
                    rf"^{re.escape(words[0])}\b", replacement, suggested_summary, count=1
                )

    return f"{suggested_type}({suggested_scope}){breaking_part}: {suggested_summary}"


def fetch_pr_commits(repo: str, pr_number: int, token: str | None = None) -> list[dict[str, str]]:
    """Retrieve all commits for a pull request via GitHub REST API with pagination."""
    commits: list[dict[str, str]] = []
    page = 1
    per_page = 100

    while True:
        url = (
            f"https://api.github.com/repos/{repo}/pulls/{pr_number}/commits"
            f"?per_page={per_page}&page={page}"
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "commit-validator",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            sys.stderr.write(
                f"Error: Failed to fetch commits from GitHub API ({err.code} {err.reason}).\n"
            )
            raise

        if not data or not isinstance(data, list):
            break

        for item in data:
            sha = item.get("sha", "")
            commit_info = item.get("commit", {})
            message = commit_info.get("message", "")
            commits.append({"sha": sha, "message": message})

        if len(data) < per_page:
            break
        page += 1

    return commits


def format_cli_report(
    failures: list[ValidationResult], pr_title_failure: ValidationResult | None
) -> str:
    """Format human-readable CLI report matching the required style."""
    output: list[str] = []

    if pr_title_failure:
        output.append("=" * 80)
        output.append("❌ PR Title Validation Failed!")
        output.append("=" * 80)
        output.append(f"PR Title: {pr_title_failure.message}")
        output.append("\nErrors:")
        for err in pr_title_failure.errors:
            output.append(f"- {err}")
        output.append("\nRequired format:")
        output.append("  TYPE(scope): summary  or  TYPE(scope)!: summary")
        output.append("\nExpected:")
        output.append(f"  {pr_title_failure.suggested}")
        output.append(
            "\nNote: PR title must follow convention to prevent non-compliant squash-merge commits."
        )
        output.append("-" * 80)
        output.append("")

    if failures:
        output.append("=" * 80)
        output.append("❌ Commit Message Validation Failed!")
        output.append("=" * 80)
        for idx, item in enumerate(failures, 1):
            short_sha = item.sha[:7] if item.sha else "unknown"
            first_line = item.message.splitlines()[0] if item.message else ""
            output.append(f"Commit: {short_sha}")
            output.append(f"Message: {first_line}")
            output.append("\nErrors:")
            for err in item.errors:
                output.append(f"- {err}")
            output.append("\nExpected:")
            output.append(f"  {item.suggested}")
            if idx < len(failures):
                output.append("-" * 80)
        output.append("=" * 80)

    return "\n".join(output)


def write_github_step_summary(
    failures: list[ValidationResult],
    pr_title_failure: ValidationResult | None,
    total_commits: int,
) -> None:
    """Write markdown summary to GITHUB_STEP_SUMMARY if available."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    md_lines: list[str] = []
    if failures or pr_title_failure:
        md_lines.append("## ❌ Commit Convention Validation Failed\n")
        md_lines.append(
            "Every commit in a Pull Request and the PR title must adhere to "
            "[`docs/commit-convention.md`](docs/commit-convention.md).\n"
        )

        if pr_title_failure:
            md_lines.append("### Pull Request Title Violation\n")
            md_lines.append(f"- **Current Title:** `{pr_title_failure.message}`")
            md_lines.append(f"- **Expected Title:** `{pr_title_failure.suggested}`")
            md_lines.append("- **Errors:**")
            for err in pr_title_failure.errors:
                md_lines.append(f"  - {err}")
            md_lines.append(
                "\n> **Why this matters:** When a PR is squash-merged, GitHub uses the PR title "
                "as the commit message on `main`. An invalid PR title will produce a non-compliant "
                "commit on `main`.\n"
            )

        if failures:
            md_lines.append(f"### Invalid Commits ({len(failures)} of {total_commits})\n")
            md_lines.append("| Commit | Message | Errors | Suggested Fix |")
            md_lines.append("| :--- | :--- | :--- | :--- |")
            for f in failures:
                short_sha = f.sha[:7] if f.sha else "HEAD"
                first_line = f.message.splitlines()[0] if f.message else ""
                err_text = "<br>".join(f"• {e}" for e in f.errors)
                md_lines.append(
                    f"| `{short_sha}` | `{first_line}` | {err_text} | `{f.suggested}` |"
                )
            md_lines.append("\n### How to fix invalid commits locally:\n")
            md_lines.append("```bash")
            md_lines.append("# If the last commit is invalid:")
            md_lines.append('git commit --amend -m "TYPE(scope): summary"')
            md_lines.append("")
            md_lines.append("# If older commits in your branch are invalid:")
            md_lines.append("git rebase -i origin/main")
            md_lines.append(
                "# Mark 'r' or 'reword' for each invalid commit and update the message."
            )
            md_lines.append("git push --force-with-lease")
            md_lines.append("```\n")
    else:
        md_lines.append("## ✅ Commit Convention Validation Passed\n")
        md_lines.append(
            f"All **{total_commits}** commit(s) and the PR title conform to the project convention."
        )

    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n" + "\n".join(md_lines) + "\n")
    except OSError as err:
        sys.stderr.write(f"Warning: Could not write to GITHUB_STEP_SUMMARY: {err}\n")


def emit_github_annotations(
    failures: list[ValidationResult], pr_title_failure: ValidationResult | None
) -> None:
    """Emit GitHub Actions error workflow commands."""
    if pr_title_failure:
        err_msg = "; ".join(pr_title_failure.errors)
        print(f"::error title=Invalid PR Title::{err_msg}. Expected: {pr_title_failure.suggested}")

    for f in failures:
        short_sha = f.sha[:7] if f.sha else "commit"
        err_msg = "; ".join(f.errors)
        print(f"::error title=Invalid Commit ({short_sha})::{err_msg}. Expected: {f.suggested}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate commit messages and PR title against project convention."
    )
    parser.add_argument("--message", help="Validate a single commit message string.")
    parser.add_argument("--pr-title", help="Validate a PR title string.")
    parser.add_argument(
        "--pr-number", type=int, help="Pull request number to validate via GitHub API."
    )
    parser.add_argument("--repo", help="Repository slug in format 'owner/repo'.")
    parser.add_argument("--token", help="GitHub Personal Access Token or GITHUB_TOKEN.")
    parser.add_argument(
        "--commits-json",
        help="Path to JSON file containing a list of objects with 'sha' and 'message'.",
    )

    args = parser.parse_args(argv)

    failures: list[ValidationResult] = []
    pr_title_failure: ValidationResult | None = None
    total_commits = 0

    # Mode 1: Direct single message validation
    if args.message is not None:
        total_commits = 1
        errors = validate_commit_message(args.message)
        if errors:
            suggested = suggest_valid_commit(args.message)
            failures.append(
                ValidationResult(sha="", message=args.message, errors=errors, suggested=suggested)
            )

    # Mode 2: Validate from JSON file
    elif args.commits_json:
        with open(args.commits_json, "r", encoding="utf-8") as f:
            commit_list = json.load(f)
        total_commits = len(commit_list)
        for c in commit_list:
            sha = c.get("sha", "")
            msg = c.get("message", "")
            errors = validate_commit_message(msg)
            if errors:
                suggested = suggest_valid_commit(msg)
                failures.append(
                    ValidationResult(sha=sha, message=msg, errors=errors, suggested=suggested)
                )

    # Mode 3: Validate via GitHub API or GitHub Actions environment
    else:
        repo = args.repo or os.getenv("GITHUB_REPOSITORY")
        pr_number = args.pr_number
        token = args.token or os.getenv("GITHUB_TOKEN")
        pr_title = args.pr_title

        event_path = os.getenv("GITHUB_EVENT_PATH")
        if event_path and os.path.exists(event_path):
            with open(event_path, "r", encoding="utf-8") as f:
                event_data = json.load(f)
            pr_data = event_data.get("pull_request", {})
            if not pr_number:
                pr_number = pr_data.get("number")
            if not pr_title:
                pr_title = pr_data.get("title")
            if not repo:
                repo = event_data.get("repository", {}).get("full_name")

        if not repo or not pr_number:
            sys.stderr.write(
                "Error: Missing repository or PR number. In GitHub Actions, ensure the workflow "
                "runs on pull_request events, or provide --repo and --pr-number locally.\n"
            )
            return 2

        # Validate PR title if available
        if pr_title:
            title_errors = validate_commit_message(pr_title, is_pr_title=True)
            if title_errors:
                suggested_title = suggest_valid_commit(pr_title, is_pr_title=True)
                pr_title_failure = ValidationResult(
                    sha="PR_TITLE",
                    message=pr_title,
                    errors=title_errors,
                    suggested=suggested_title,
                )

        # Fetch and validate all PR commits
        try:
            commits = fetch_pr_commits(repo, pr_number, token)
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            OSError,
        ) as err:
            sys.stderr.write(f"Failed to fetch PR commits: {err}\n")
            return 2

        total_commits = len(commits)
        for item in commits:
            sha = item["sha"]
            msg = item["message"]
            errors = validate_commit_message(msg)
            if errors:
                suggested = suggest_valid_commit(msg)
                failures.append(
                    ValidationResult(sha=sha, message=msg, errors=errors, suggested=suggested)
                )

    # Also check standalone --pr-title if given in mode 1
    if args.pr_title and not pr_title_failure:
        title_errors = validate_commit_message(args.pr_title, is_pr_title=True)
        if title_errors:
            suggested_title = suggest_valid_commit(args.pr_title, is_pr_title=True)
            pr_title_failure = ValidationResult(
                sha="PR_TITLE",
                message=args.pr_title,
                errors=title_errors,
                suggested=suggested_title,
            )

    # Reporting
    if failures or pr_title_failure:
        report = format_cli_report(failures, pr_title_failure)
        print(report)
        emit_github_annotations(failures, pr_title_failure)
        write_github_step_summary(failures, pr_title_failure, total_commits)
        return 1

    print(f"✅ Success: All {total_commits} commit(s) and PR title adhere to convention.")
    write_github_step_summary([], None, total_commits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
