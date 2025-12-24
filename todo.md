# Project TODO — GPL Compliance & Repo Hygiene

This checklist tracks the steps needed to properly license and document this
project under the GNU General Public License v3.0 or later (GPL-3.0-or-later),
across all language ecosystems used in this repository.

---

## Licensing (Root)

- [ ] Choose GPL variant: **GPL-3.0-or-later**
- [ ] Add `LICENSE` (full GPLv3 text, unmodified)
- [ ] Add licensing paragraph to `README.md`
- [ ] Add `LICENSING.md` clarifying scope, derivatives, and FMUs
- [ ] Add `CITATION.cff` for academic citation
- [ ] Add `NOTICE` (placeholder for third-party notices)

---

## Python Ecosystem

- [ ] Add `pyproject.toml` with GPL license metadata
- [ ] Verify `requirements.txt` is present and complete
- [ ] Add SPDX header to **every** `.py` file
- [ ] Add license docstring to package-level `__init__.py`
- [ ] Confirm all dependencies are GPL-compatible

---

## Modelica Ecosystem

- [ ] Add GPL documentation annotation to **every** `package.mo`
- [ ] Verify all Modelica subpackages inherit or repeat license info
- [ ] Confirm FMU export includes `resources/LICENSE.txt`
- [ ] (Optional) Add license note to `modelDescription.xml`

---

## Go Ecosystem (Future)

- [ ] Add GPL file headers to all `.go` files
- [ ] Verify `go.mod` exists and repo-level `LICENSE` is referenced
- [ ] Add license text to any distributed Go binaries

---

## Binaries & Artifacts

- [ ] Ensure FMUs include license text
- [ ] Ensure distributed scripts or tools reference GPL
- [ ] Ensure generated outputs (plots, PDFs) are **not** marked as GPL

---

## Final Verification

- [ ] Spot-check random files for missing headers
- [ ] Confirm no conflicting licenses are introduced
- [ ] Verify GitHub detects license correctly
- [ ] Tag initial GPL-compliant release

---

## Done Criteria

This checklist is complete when:
- GPL intent is unambiguous in every language ecosystem
- License propagation is explicit and visible
- No reasonable user can be confused about reuse terms
