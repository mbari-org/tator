"""
Database URI config: YAML-only configuration for per-project MongoDB URIs and FiftyOne app ports.
See database_uris.example.yaml for structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


def database_name_from_uri(uri: str) -> str:
    """Extract MongoDB database name from URI (path component). Default 'fiftyone' if missing."""
    from urllib.parse import urlparse
    p = urlparse(uri)
    name = (p.path or "").strip("/")
    return name or "fiftyone"


@dataclass(frozen=True)
class DatabaseEntry:
    """Single database: URI and FiftyOne app port."""

    uri: str
    port: int


@dataclass
class ProjectConfig:
    """Per-project config: optional vss_project and list of databases (uri/port). Key is project name."""

    vss_project: str | None = None
    databases: list[DatabaseEntry] = field(default_factory=list)


@dataclass
class DatabaseUriConfig:
    """
    Root config: projects keyed by project name -> ProjectConfig.
    Load from YAML only via from_yaml_path() or from_yaml_string().
    """

    projects: dict[str, ProjectConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml_path(cls, path: str | Path) -> DatabaseUriConfig:
        """Load config from a YAML file. Only .yaml/.yml paths accepted (no JSON)."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix not in (".yaml", ".yml"):
            raise ValueError(
                f"Database URI config must be YAML; path must end with .yaml or .yml, got: {path!r}"
            )
        if not path.is_file():
            raise FileNotFoundError(f"Database URI config file not found: {path!r}")
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_yaml_string(f.read())

    @classmethod
    def from_yaml_string(cls, content: str) -> DatabaseUriConfig:
        """Load config from a YAML string. JSON is not accepted."""
        raw = yaml.safe_load(content)
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(
                f"Database URI config YAML must be a mapping at top level, got {type(raw).__name__}"
            )
        projects_raw = raw.get("projects")
        if projects_raw is None:
            return cls()
        if not isinstance(projects_raw, dict):
            raise ValueError(
                f"config 'projects' must be a mapping, got {type(projects_raw).__name__}"
            )
        projects: dict[str, ProjectConfig] = {}
        for key, proj in projects_raw.items():
            if not isinstance(proj, dict):
                raise ValueError(
                    f"project {key!r} must be a mapping, got {type(proj).__name__}"
                )
            vss_project = proj.get("vss_project")
            if vss_project is not None and not isinstance(vss_project, str):
                raise ValueError(
                    f"project {key!r} 'vss_project' must be a string if present, got {type(vss_project).__name__}"
                )
            db_list_raw = proj.get("databases")
            if db_list_raw is None:
                db_list_raw = []
            if not isinstance(db_list_raw, list):
                raise ValueError(
                    f"project {key!r} 'databases' must be a list, got {type(db_list_raw).__name__}"
                )
            entries: list[DatabaseEntry] = []
            for i, db in enumerate(db_list_raw):
                if not isinstance(db, dict):
                    raise ValueError(
                        f"project {key!r} databases[{i}] must be a mapping, got {type(db).__name__}"
                    )
                uri = db.get("uri")
                port = db.get("port")
                if not isinstance(uri, str):
                    raise ValueError(
                        f"project {key!r} databases[{i}] 'uri' must be a string, got {type(uri).__name__}"
                    )
                if not isinstance(port, int):
                    raise ValueError(
                        f"project {key!r} databases[{i}] 'port' must be an int, got {type(port).__name__}"
                    )
                entries.append(DatabaseEntry(uri=uri, port=port))
            projects[str(key)] = ProjectConfig(
                vss_project=vss_project, databases=entries
            )
        return cls(projects=projects)
