# Contributors

Copyright (c) 2025-2026, Science and Technology Facilities Council (STFC),
UKRI, and The University of Sheffield; see `LICENSE`.

The toolkit is developed under the [CCP-NTH](https://ccpnth.ac.uk/) project
(Collaborative Computational Project in Nuclear Thermal Hydraulics).

## Authors

| Name | Affiliation | Role |
|---|---|---|
| **Alex Old** | PhD student, The University of Sheffield | Original and principal author |
| **Wei Wang** | Principal Computational Scientist, Science and Technology Facilities Council (STFC), UKRI | Co-author and maintainer |

**Alex Old** designed and wrote the toolkit as part of his PhD, and wrote
most of the code.

**Wei Wang** co-supervised Alex's PhD project, under which the toolkit was
developed. She added support for cylindrical geometries (pipe and annulus),
adapted the toolkit to the current CHAPSim2 output layout, ported the
solver's fluid property model, and added case consistency checking, figure
provenance, the package layout, the test suite and continuous integration.

## Contributing

Contributions are welcome through issues and pull requests on
[GitHub](https://github.com/weiwangstfc/CHAPSim2-toolkit). By contributing you
agree that your contribution is licensed under the project's BSD 3-Clause
licence. If you contribute, add yourself to the table above in the same pull
request, and to `CITATION.cff` if you would like to be cited.

Before opening a pull request, run the tests (`python run_tests.py`) and add an
entry to `CHANGELOG.md`, marked **behaviour change** if it alters a number
someone may already have published.
