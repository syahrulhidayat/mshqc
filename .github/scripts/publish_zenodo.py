#!/usr/bin/env python3
"""
Automate Zenodo deposition creation/versioning, file upload (Bucket API), and publishing
from GitHub Actions using ZENODO_TOKEN.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


def _headers(token: str, content_type: Optional[str] = None) -> Dict[str, str]:
    hdr = {"Authorization": f"Bearer {token}"}
    if content_type:
        hdr["Content-Type"] = content_type
    return hdr


def _raise_for_status(resp: requests.Response, ctx: str) -> None:
    if not resp.ok:
        raise RuntimeError(
            f"[Zenodo Error] {ctx} failed ({resp.status_code}): {resp.text[:1200]}"
        )


def _find_existing_version(
    base_url: str, token: str, title_prefix: str, version: str
) -> Optional[Dict[str, Any]]:
    resp = requests.get(
        f"{base_url}/deposit/depositions",
        headers=_headers(token),
        params={"size": 50},
        timeout=30,
    )
    _raise_for_status(resp, "list depositions")
    for dep in resp.json():
        meta = dep.get("metadata", {})
        dep_ver = str(meta.get("version", "")).lstrip("v")
        dep_title = str(meta.get("title", ""))
        if dep_ver == version.lstrip("v") and dep_title.lower().startswith(
            title_prefix.lower()
        ):
            return dep
    return None


def _create_or_version_deposition(
    base_url: str, token: str, deposit_id: str
) -> Dict[str, Any]:
    if deposit_id:
        resp = requests.post(
            f"{base_url}/deposit/depositions/{deposit_id}/actions/newversion",
            headers=_headers(token),
            timeout=60,
        )
        _raise_for_status(resp, f"newversion for {deposit_id}")
        draft_url = resp.json()["links"]["latest_draft"]
        draft_resp = requests.get(draft_url, headers=_headers(token), timeout=30)
        _raise_for_status(draft_resp, "fetch latest_draft")
        draft = draft_resp.json()
        for old_file in draft.get("files", []):
            del_url = old_file.get("links", {}).get("self")
            if del_url:
                requests.delete(del_url, headers=_headers(token), timeout=30)
        return draft

    resp = requests.post(
        f"{base_url}/deposit/depositions",
        headers=_headers(token, "application/json"),
        json={},
        timeout=30,
    )
    _raise_for_status(resp, "create empty deposition")
    return resp.json()


def _upload_file_bucket(
    bucket_url: str, token: str, file_path: Path
) -> None:
    url = f"{bucket_url}/{file_path.name}"
    with file_path.open("rb") as fh:
        resp = requests.put(
            url,
            data=fh,
            headers=_headers(token, "application/octet-stream"),
            timeout=300,
        )
    _raise_for_status(resp, f"upload {file_path.name}")
    print(f"[Zenodo] Uploaded: {file_path.name} ({file_path.stat().st_size / 1e6:.2f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish mshqc release artifacts to Zenodo")
    parser.add_argument("--dist-dir", default="dist", help="Directory containing .tar.gz, .whl, SHA256SUMS.txt")
    parser.add_argument("--metadata", default=".zenodo.json", help="Path to .zenodo.json")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        sys.exit("Error: ZENODO_TOKEN tidak ditemukan di environment / GitHub Secrets.")

    sandbox = os.environ.get("ZENODO_SANDBOX", "false").lower() == "true"
    base_url = "https://sandbox.zenodo.org/api" if sandbox else "https://zenodo.org/api"
    deposit_id = os.environ.get("ZENODO_DEPOSIT_ID", "").strip()
    version = os.environ.get("RELEASE_VERSION", "2.0.0").strip()
    tag = os.environ.get("RELEASE_TAG", f"v{version}").strip()
    repo = os.environ.get("GITHUB_REPOSITORY", "syahrulhidayat/mshqc").strip()

    meta_path = Path(args.metadata)
    if not meta_path.is_file():
        sys.exit(f"Error: File metadata tidak ditemukan: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    existing = _find_existing_version(base_url, token, "mshqc", version)
    if existing and existing.get("submitted"):
        doi = existing.get("doi") or existing.get("metadata", {}).get("doi", "N/A")
        print(f"[Zenodo] Version {version} sudah dipublikasikan sebelumnya (DOI: {doi}). Melewati upload.")
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary_file:
            with open(summary_file, "a", encoding="utf-8") as sf:
                sf.write(f"\n### Zenodo Archive (Already Published)\n- **Version:** `{version}`\n- **DOI:** `{doi}`\n")
        return

    dist_dir = Path(args.dist_dir)
    artifacts: List[Path] = sorted(
        list(dist_dir.glob("*.tar.gz"))
        + list(dist_dir.glob("*.whl"))
        + list(dist_dir.glob("SHA256SUMS.txt"))
    )
    if not artifacts:
        sys.exit(f"Error: Tidak ada artefak rilis di {dist_dir}")

    dep = _create_or_version_deposition(base_url, token, deposit_id)
    dep_id = dep["id"]
    bucket_url = dep["links"]["bucket"]
    print(f"[Zenodo] Draft deposition ready: ID={dep_id} (sandbox={sandbox})")

    for fp in artifacts:
        _upload_file_bucket(bucket_url, token, fp)

    meta["version"] = version
    meta["publication_date"] = dt.date.today().isoformat()
    related = list(meta.get("related_identifiers", []))
    gh_tree_url = f"https://github.com/{repo}/tree/{tag}"
    if not any(r.get("identifier") == gh_tree_url for r in related):
        related.append(
            {
                "identifier": gh_tree_url,
                "relation": "isSupplementTo",
                "scheme": "url",
            }
        )
    meta["related_identifiers"] = related

    put_resp = requests.put(
        f"{base_url}/deposit/depositions/{dep_id}",
        headers=_headers(token, "application/json"),
        json={"metadata": meta},
        timeout=30,
    )
    _raise_for_status(put_resp, "update metadata")

    pub_resp = requests.post(
        f"{base_url}/deposit/depositions/{dep_id}/actions/publish",
        headers=_headers(token),
        timeout=60,
    )
    _raise_for_status(pub_resp, "publish deposition")
    published = pub_resp.json()
    doi = published.get("doi", "N/A")
    concept_doi = published.get("conceptdoi", "N/A")
    record_url = published.get("links", {}).get("record_html", "")

    print(f"[Zenodo] PUBLISHED! DOI={doi} | ConceptDOI={concept_doi} | URL={record_url}")
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_file:
        with open(summary_file, "a", encoding="utf-8") as sf:
            sf.write(
                f"\n### Zenodo Release Archive\n"
                f"- **Version:** `{version}` (`{tag}`)\n"
                f"- **Version DOI:** `{doi}`\n"
                f"- **Concept DOI:** `{concept_doi}`\n"
                f"- **Record URL:** {record_url}\n"
            )


if __name__ == "__main__":
    main()
