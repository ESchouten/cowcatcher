import argparse
import json
from pathlib import Path

from aidetector.dazzlecow.enrollment import (
    DEFAULT_ENROLLMENT_MARGIN,
    DEFAULT_ENROLLMENT_SIMILARITY,
    finalize_enrollment,
    finalize_pending_enrollment,
)
from aidetector.dazzlecow.tracklet_store import TrackletStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage cow identity enrollment")
    parser.add_argument("--database", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    finalize = commands.add_parser("finalize")
    finalize.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_ENROLLMENT_SIMILARITY,
    )
    finalize.add_argument(
        "--margin-threshold",
        type=float,
        default=DEFAULT_ENROLLMENT_MARGIN,
    )
    finalize.add_argument("--identity-count", type=int)

    commands.add_parser("list")
    name = commands.add_parser("name")
    name.add_argument("identity")
    name.add_argument("animal_number")

    arguments = parser.parse_args()
    with TrackletStore(arguments.database) as store:
        if arguments.command == "finalize":
            assignments = (
                finalize_pending_enrollment(
                    store,
                    similarity_threshold=arguments.similarity_threshold,
                    margin_threshold=arguments.margin_threshold,
                )
                if store.is_finalized()
                else finalize_enrollment(
                    store,
                    similarity_threshold=arguments.similarity_threshold,
                    margin_threshold=arguments.margin_threshold,
                    identity_count=arguments.identity_count,
                )
            )
            print(
                json.dumps(
                    {
                        "tracklets": len(assignments),
                        "identities": len(set(assignments.values())),
                    },
                    indent=2,
                )
            )
        elif arguments.command == "name":
            store.set_animal_number(arguments.identity, arguments.animal_number)
        else:
            print(json.dumps(store.identities(), indent=2))
