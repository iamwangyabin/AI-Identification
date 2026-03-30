from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SampleRecord:
    path: str
    key: str
    label: int
    class_name: str | None


def build_path_substitutions(specs: Iterable[str]) -> list[tuple[str, str]]:
    substitutions: list[tuple[str, str]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid path substitution '{spec}'. Expected FROM=TO format."
            )
        source, target = spec.split("=", 1)
        substitutions.append((source, target))
    return substitutions


class JsonlImageDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        manifest_path: str | Path,
        transform: Callable | None = None,
        data_root: str | Path | None = None,
        path_substitutions: list[tuple[str, str]] | None = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.data_root = Path(data_root) if data_root is not None else None
        self.path_substitutions = path_substitutions or []
        self.samples = self._load_samples()

    def _apply_path_substitutions(self, path: str) -> str:
        for source, target in self.path_substitutions:
            if path.startswith(source):
                return target + path[len(source) :]
        return path

    def _resolve_path(self, path: str) -> str:
        raw_path = Path(path)
        if raw_path.is_absolute():
            return self._apply_path_substitutions(str(raw_path))
        if self.data_root is not None:
            return str((self.data_root / raw_path).resolve())
        return str((self.manifest_path.parent / raw_path).resolve())

    def _load_samples(self) -> list[SampleRecord]:
        samples: list[SampleRecord] = []
        if self.manifest_path.suffix.lower() == ".csv":
            with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for record in reader:
                    path = record.get("path") or record.get("file_name")
                    if path is None:
                        raise ValueError(
                            f"CSV manifest '{self.manifest_path}' must contain a 'path' or 'file_name' column."
                        )
                    samples.append(
                        SampleRecord(
                            path=self._resolve_path(path),
                            key=record.get("key", Path(path).name),
                            label=int(record["label"]),
                            class_name=record.get("class_name"),
                        )
                    )
        else:
            with self.manifest_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    path = record.get("path") or record.get("file_name")
                    if path is None:
                        raise ValueError(
                            f"JSONL manifest '{self.manifest_path}' must contain a 'path' or 'file_name' field."
                        )
                    samples.append(
                        SampleRecord(
                            path=self._resolve_path(path),
                            key=record.get("key", Path(path).name),
                            label=int(record["label"]),
                            class_name=record.get("class_name"),
                        )
                    )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample.label,
            "key": sample.key,
            "path": sample.path,
            "class_name": sample.class_name,
        }
