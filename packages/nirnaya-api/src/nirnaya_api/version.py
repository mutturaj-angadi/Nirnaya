"""Single source of truth for the package version.

Kept as a standalone module (no other nirnaya_api imports) so that
``nirnaya version`` and the REST ``/version`` endpoint can report it
cheaply and so build tooling can read it without importing the whole
package (and its optional heavy dependencies).
"""

__version__ = "0.1.0"

#: Bumped whenever the JSON model format (docs/JSON_FORMAT.md) changes
#: in a backward-incompatible way. Included in API/CLI output so callers
#: can detect drift independent of the package version.
JSON_MODEL_FORMAT_VERSION = "1.0"

#: Bumped whenever the REST wire schemas change in a backward-incompatible
#: way.
API_SCHEMA_VERSION = "1.0"
