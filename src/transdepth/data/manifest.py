"""Safe JSONL manifest serialization and role gates."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from transdepth.data.schema import SampleRecord


def write_manifest(records: list[SampleRecord], path: str | Path) -> dict[str, object]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ids = [item.sample_id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest sample IDs must be unique")
    digest = hashlib.sha256()
    with NamedTemporaryFile("w", encoding="utf-8", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        for record in records:
            line = json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
            stream.write(line)
            digest.update(line.encode())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return {
        "path": str(destination),
        "sha256": digest.hexdigest(),
        "records": len(records),
        "roles": dict(sorted(Counter(item.assigned_role for item in records).items())),
    }


def read_manifest(path: str | Path, allowed_roles: set[str] | None = None) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                record = SampleRecord.from_dict(json.loads(line))
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid manifest record at line {line_number}") from error
            if allowed_roles is not None and record.assigned_role not in allowed_roles:
                continue
            records.append(record)
    ids = [item.sample_id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("manifest contains duplicate sample IDs")
    return records
