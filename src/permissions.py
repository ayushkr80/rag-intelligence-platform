"""Permission policy — enforced inside the retrieval layer, not the app layer.

App-layer filtering leaks: by the time the application sees results, restricted
documents already consumed top-k slots and their content reached the model.
Filtering the candidate pool inside retrieval prevents both.
"""

ROLES = {
    "analyst": {"public"},
    "employee": {"public", "internal", "confidential"},
}

DEFAULT_LEVEL = "public"


def permitted_indices(metadatas: list[dict | None], role: str) -> set[int]:
    """Return indices of chunks the role may see."""
    allowed_levels = ROLES.get(role, {"public"})
    return {
        index
        for index, meta in enumerate(metadatas)
        if meta and meta.get("level", DEFAULT_LEVEL) in allowed_levels
    }
