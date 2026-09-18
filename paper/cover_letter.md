Dear Professor Given,

I submit "Downloads Are Cheap: Validated Use as a Measure of Research Software" for consideration as a research article in the *Journal of the Association for Information Science and Technology*.

Credit for research software rests on counts that cost little to produce. A download is one fetch of a file, and build servers, mirrors and frequent releases multiply it; a mention in a methods section is often missing. The paper proposes a count that costs a published paper to move: whether a package is loaded in the replication code a journal requires and checks. I measure it by parsing every R, Python and Stata file in 13,245 deposits from 75 economics and political science journal collections, which load 4,565 packages.

Three results follow.

1. Download counts and validated use agree loosely. Among packages the corpus uses, the rank correlation is 0.61 for Stata, 0.53 for R and 0.36 for Python, and 0.31 among R packages in at least twenty deposits. `stargazer` is the fourth most used R package and the 388th most downloaded. Downloads correlate at 0.87 with the number of packages that install a package as a dependency, so they come close to measuring its place in the dependency graph. Holding the audience roughly fixed, with CRAN's Econometrics and Causal Inference task views, raises the agreement to 0.65.
2. The most used software prepares tables and figures. `estout` is in 43% of deposits containing Stata, ahead of every estimator.
3. Stata appears in 1.4 times as many deposits as R and has been left out of studies of research software because no public index mapped a Stata command to its package. Its packages are spread across SSC, the *Stata Journal*, the *Stata Technical Bulletin* and authors' own sites. I built that index, 10,147 commands in 5,581 packages, and released it.

The paper continues the line of work in this journal on how software appears in the scholarly record (Howison and Bullard 2016; Du, Cohoon, Lopez and Howison 2021) and adds an audit of what the standard count of software use measures. The mention-level data (14.9 million rows), the aggregate tables and the Stata index are deposited under CC0 at 10.5281/zenodo.21943908 and 10.5281/zenodo.21926099, and the code is at https://github.com/recite/softverse.

The manuscript is not under review elsewhere. I have no competing financial interests. The argument that better counts would increase the production of research software is my own, from earlier writing that the paper cites.

Suggested reviewers:

- [name, affiliation, email]
- [name, affiliation, email]
- [name, affiliation, email]

Sincerely,

Gaurav Sood
[affiliation]
[email]

<!-- Numbers typed from build/check_paper.log on 2026-09-17, extractor 2.4.0. The
letter is not rendered through the paper's pipeline; recheck them against the
PDF before sending. -->
