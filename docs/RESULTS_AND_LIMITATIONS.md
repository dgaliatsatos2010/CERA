# Scope of evaluated performance

The archived benchmark comprises 12 signal and 12 unimodal-null generating
families, each tested at n=240 and n=480 with two replications (96 datasets).
It is an **exploratory synthetic study**, not a representative guarantee.

CERA's observed average ARI on the 48 signal datasets was about **0.4753**
(16/48 with exact generator-defined group count). It generated no operational
false splits on the 48 null datasets under this study's null definition.
These observations are **not** proof of nominal type-I error control.

Performance was poor on several oblique, overlapping and nonlinear families.
The study did not establish broad statistical superiority over Gaussian
mixture modeling or graph/density alternatives. Source-reported timings can
be environment-specific and are not portable speed benchmarks.

The available pilot UniForCE run used a minimum-size threshold larger than
the known minority group and is not evidence of a fair general comparison.
The GAUFS diagnostic attempts did not produce a valid cluster assignment
before the time budget expired and must not be represented as valid scores.

The full benchmark data and frozen reference results are not included in this source-only initial upload. Reproduction against the archived benchmark requires the separate full reproducibility dataset.
