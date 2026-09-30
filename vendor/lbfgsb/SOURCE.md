# Pinned community lbfgsb 0.1.1: build-only compatibility patch

Owner: Bullet ML maintainers.

Origin: published `lbfgsb 0.1.1` crate, archive SHA-256
`70b0d1a3440f0ad7ec488c97ccd03e1d76f0114342be1a5fa17442b0b25e075f`.
The published archive's VCS metadata is dirty, so a Git revision is not treated
as a clean source identity. `SOURCE.json` records hashes of every retained
upstream file. The Rust wrapper, native numerical sources/header and licenses
are unmodified. Unused Matlab bindings/examples, executables and developer
metadata are omitted. Both BSD license notices are retained.

The sole `build.rs` adaptation is a bindgen file allowlist for `lbfgsb.h`.
Without it, bindgen 0.65.1 generates bindings for all transitively included
system declarations and panics on glibc 2.39 AArch64 vector-PCS math functions
(`Invalid or unknown abi 16`, `_ZGVnN4vv_atan2f`). None of those math APIs belong
to the Rust solver interface. The filter keeps all solver-header declarations
and their required types. It changes neither native C compilation nor the
solver algorithm, stopping rules, floating-point flags or ABI types.

The root Cargo patch applies the same filter on all supported targets. No
build-time environment-variable workaround, runtime configuration or source
mutation in Cargo's shared cache is required.

Verify retained files:

```sh
(cd vendor/lbfgsb && sha256sum --check SHA256SUMS) # Linux CI
(cd vendor/lbfgsb && shasum -a 256 --check SHA256SUMS) # macOS
```

Remove this compatibility copy and the root Cargo patch/exclusion once a pinned
upstream release constrains bindings or handles AArch64 vector PCS correctly.
Before removal, run Linux x86_64, Linux ARM64 and macOS ARM64 linear-only plus
all-feature tests, the fixed-tolerance sklearn oracle, and exact artifact/OOS
reload tests. Confirm no algorithm/parameter fallback and retain the new source
identity. If updating the numerical solver too, make that a separately reviewed
change with new evidence rather than attributing it to this build fix.
