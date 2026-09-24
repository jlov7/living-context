# Security and support

This package is a process-local research reference. The catalog writer and
Boolean-only `checks_passed` admission path trust caller input and do not
inspect artifact bytes. The optional `verify_artifacts`
path checks copied artifact bytes against supplied digests, checks declared
source revisions against the staged revisions, and binds its receipt to the
exact staged candidate. These checks establish consistency of the supplied
bytes and revisions; they do not authenticate the caller or prove that training
used the claimed sources. The library does not
isolate model workers at the OS level, persist transactions, serve untrusted
tenants, or provide distributed consistency. Do not place private, employer,
customer, or patient data in the synthetic research flows.

Do not send sensitive details to a public issue or discussion. If the
[repository's private vulnerability reporting](https://github.com/jlov7/living-context/security/advisories/new)
is available, use it for a confidential report. If it is unavailable, open a
nonsensitive request for a private contact route without including exploit
details. Report ordinary defects in the
[issue tracker](https://github.com/jlov7/living-context/issues) with a minimal
reproducer, affected version, and source commit. Availability of a private
reporting route must be checked on the host; this document does not attest to
its configuration.

Supported model-free source checks target Python 3.11 and 3.13. The portable
CI declaration covers Ubuntu and macOS; its hosted status must be reported
from actual job receipts. Apple Silicon/MLX model experimentation is separate
from the portable core and is not a supported hosted CI gate.
