# Changelog and version policy

## 0.1.0 — local release candidate, 2026-09-23

- Align the package with its supported in-memory reference and valid negative
  v2 soft-prefix result.
- Add an installed-package lifecycle example and explicit admission trust
  contract.
- Add an exact full-result v2 release replay over the distributed text
  evidence. Retain the archived v1 custody check separately.
- Add source-candidate documentation, citation and security status, and a
  declared portable CI matrix. Hosted execution was unobserved at this local
  candidate checkpoint.
- Reject a document listed for both update and deletion before staging a
  candidate. Preserve the v3 revision saved-output replay against its pinned
  historical catalog source as the current runtime changes.

Versions before 1.0 may change Python API signatures; each candidate records
the change here. Frozen historical evidence and protocols are not rewritten to
match current code. A changed study requires a new protocol and result identity.
The release version is the `pyproject.toml` version, cross-checked with
`living_context.__version__` and `CITATION.cff`. A local candidate is not a
published release.
