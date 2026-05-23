from __future__ import annotations

import shlex


DATASET_BUILDER_ROOT_TEMPLATE = "{{ var.value.get('DATASET_BUILDER_ROOT', '/opt/airflow/dataset-builder') }}"


def build_bash_command(script_relative_path: str, *args: str) -> str:
    command = ["python", script_relative_path, *args]
    return " && ".join(
        [
            "set -euo pipefail",
            f"cd {DATASET_BUILDER_ROOT_TEMPLATE}",
            " ".join(shlex.quote(part) for part in command),
        ]
    )