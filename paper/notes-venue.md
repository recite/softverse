# Venue lessons: how JASIST-style empirical papers are built

Read on 2026-09-15. Word counts below were computed from machine-readable full text
(PMC JATS XML; `pdftotext -layout` on author PDFs), not eyeballed — so the section
proportions are measured, not impressions. Where a number could not be verified it is
marked **[unverified]**.

Access note up front: the Softcite JASIST article is **closed** (Unpaywall
`oa_status: "closed"`, `has_repository_copy: false`; no HAL/arXiv/ScholarWorks copy; Wiley,
ACM DL, HAL and DiVA all return 403/blocked from this machine). I could not read its full
text, so its card is reconstructed from the published abstract, indexing metadata, and the
authors' own pre-submission full draft in the project repo. Every other card is from full
text I actually read.

---

## Paper 1 — Softcite dataset (the closest topical analogue; only partially readable)

- **Title:** Softcite dataset: A dataset of software mentions in biomedical and economic research publications
- **Authors:** Caifan Du, Johanna Cohoon, Patrice Lopez, James Howison
- **Year / venue:** 2021, JASIST 72(7), 870–884 (15 journal pages)
- **DOI / URL:** 10.1002/asi.24454 — https://doi.org/10.1002/asi.24454
- **Abstract (verbatim, 104 words):** "Software contributions to academic research are relatively invisible, especially to the formalized scholarly reputation system based on bibliometrics. In this article, we introduce a gold-standard dataset of software mentions from the manual annotation of 4,971 academic PDFs in biomedicine and economics. The dataset is intended to be used for automatic extraction of software mentions from PDF format research publications by supervised learning at scale. We provide a description of the dataset and an extended discussion of its creation process, including improved text conversion of academic PDFs. Finally, we reflect on our challenges and lessons learned during the dataset creation, in hope of encouraging more discussion about creating datasets for machine learning use."
- **Abstract style:** one unstructured paragraph, short for the venue (~104 words vs 190–240 in the other three). Sentence 1 is the *problem in the world* ("software contributions… are relatively invisible"), sentence 2 is the *artifact and its size*, sentence 3 the *intended use*, sentences 4–5 the *two secondary contributions* (process description, reflection). No results numbers at all — because the contribution is a resource, not a finding.
- **Section structure:** **[unverified]** for the published version. The authors' own full draft (`docs/papers/dataset_description/meta_version_full_text.md`, 7,940 words, in github.com/howisonlab/softcite-dataset) is explicitly a three-part design and the published abstract confirms all three parts survived: "The goal of this paper is thus two-fold: to present a labeled dataset, and to reflect on the process and challenges in its creation and presentation… The first is a paper within a paper, straightforwardly describing our annotation project and its results. The second provides substantial process and provenance description (including mis-steps and detailed decisions). The third reflects on the overall process, describing tensions experienced…". Draft headings: *Genre paper* (motivations and prior work → prototype use for supervised ML → future work) / *Creation of the dataset* (genesis, selection of papers, collaboration infrastructure, assessing agreement) / *Connecting with GROBID for full-text alignment* (iterative refinement, alignment, post-alignment agreement calculation, consistency review, guidance in the consistency phase, provenance vs usefulness) / *Conclusion*.
- **Methods/validation vs findings:** in the draft the whole middle and back half is process and agreement measurement; the "results" are the dataset's descriptive statistics plus a demonstration of use (a supervised model trained on it). This is the one genre where validation legitimately outweighs findings — *and the paper is framed accordingly from the title and abstract onward* ("a dataset of…", "The dataset is intended to be used for…").
- **Dataset release:** dataset is a TEI/XML corpus on Zenodo (10.5281/zenodo.4444074) plus a live GitHub repo (howisonlab/softcite-dataset) with a data dictionary and setup instructions; the paper's selling point is the process/provenance description, not just the download link.
- **References:** indexes disagree — OpenAlex 62, Semantic Scholar 73, CoLab 44, OUCI (DOI-only) 32. Best estimate **~60–70**. Either way, an order of magnitude more than 8.
- **Tables/figures:** **[unverified]**.
- **Limitations:** handled as a first-class contribution ("reflect on our challenges and lessons learned"), not a defensive paragraph.

## Paper 2 — Schindler, Bensmann, Dietze & Krüger (2022) — read in full

- **Title:** The role of software in science: a knowledge graph-based analysis of software mentions in PubMed Central
- **Year / venue / DOI:** 2022, PeerJ Computer Science 8:e835, 10.7717/peerj-cs.835 — https://pmc.ncbi.nlm.nih.gov/articles/PMC8771769/
- **Abstract:** 240 words, one unstructured paragraph, moves problem → gap ("missing rigor in software citation practices renders the automated detection and disambiguation of software mentions a challenging problem") → what was done → **hard numbers in the abstract** ("11.8 M software mentions… more than 300 M triples", "more than 3 million scientific articles") → release promise ("all data and models are shared publicly").
- **Section structure with measured lengths (body total 19,034 words):**
  | Section | Words | Share |
  |---|---|---|
  | Introduction | 940 | 4.9% |
  | Related Work (2 subsections) | 1,993 | 10.5% |
  | Methods and Materials | 5,500 | 28.9% |
  | Results: Information Extraction **Performance** (= validation) | 1,670 | 8.8% |
  | Results: Analysis of Software Mentions (= findings) | 4,241 | 22.3% |
  | Discussion (incl. *Limitations of the study* 536) | 3,794 | 19.9% |
  | Conclusion (incl. *Software and Data* 284) | 874 | 4.6% |
- **First two paragraphs:** ¶1 is four sentences of stakes with zero citations — "Science across all disciplines has become increasingly data-driven… transparency about software used as part of the scientific process is crucial to ensure reproducibility". ¶2 raises the altitude to why anyone should care at scale ("from a macro-perspective, understanding software usage, varying citation habits and their evolution over time… can shape the understanding of the evolution of scientific disciplines") and only then names prior datasets. The **contribution is not in ¶1–2**; it lands in ¶7 as a labelled bulleted list ("In summary, our contributions include: *A large-scale analysis of software usage* across 3,215,386 scholarly publications…"), followed by ¶9, a literal roadmap paragraph ("The remaining paper is organized as follows…").
- **Related work:** its own section, 10.5% of the body, placed after the Introduction; split into "Requirements for large scale software citation analyses" (770) and "Previous analyses of software in scholarly publication" (1,221) — i.e. one subsection derives *criteria the paper must meet*, the other reviews *what others found*.
- **Validation vs findings:** validation is quarantined into its own Results section and is **2.5× smaller** than the findings section (1,670 vs 4,241). Method-building takes the Methods section; how well it worked takes 9% of the paper; what it bought takes 22%.
- **Limitations:** a named subsection inside the Discussion, 536 words, and unusually candid about error propagation: "the given evaluation for software and mention type classification does take error propagation into account, but the results for RE and entity disambiguation do not. Therefore, the F = 0.94 performance for RE might overestimate the true performance as it relies on results of F = 0.885 entity recognition." It also names the sampling frame bias first ("the selection of PMC as primary data source implies a bias towards Medicine").
- **Dataset release:** a dedicated *Software and Data* subsection inside the Conclusion listing every dependency with version numbers ("Python 3.9.5… PyTorch 1.9.0… R 4.1.1"), the analysis repo (github.com/f-krueger/SoftwareKG-PMC-Analysis) and the Zenodo DOI for the graph.
- **Appendix/supplementary:** one supplemental file, "Appendix Tables and Figure" — 11 of the 14 tables are appendix tables (hyper-parameter grids, domain categorisation). The main text keeps only the tables a reader must see.
- **References:** 55. **Figures:** 15. **Tables:** 14.

## Paper 3 — Thelwall & Jiang (2025), JASIST — read in full

- **Title:** Is OpenAlex suitable for research quality evaluation and which citation indicator is best?
- **Year / venue / DOI:** 2025, JASIST, 10.1002/asi.70020; accepted manuscript read at https://arxiv.org/abs/2502.18427 (29 double-spaced pages, 8,738 words to the reference list)
- **Abstract:** 219 words, **quasi-structured with inline labels** — "This article compares (1)… and (2)… **Methods (1&2):** the indicators calculated from 28.6 million articles were compared through 8,704 correlations… **Results:** (1) OpenAlex provides better citation counts than Scopus…". Numbers in the abstract; the counterintuitive finding is stated bluntly ("Counterintuitively, raw citation counts are at least as good as nearly all field normalised indicators").
- **Section structure with measured lengths:**
  | Section | Words | Share |
  |---|---|---|
  | Introduction (4 sentences) + *OpenAlex* 299 + *Citation-based indicators* 713 + *Research questions* 268 | 1,329 | 15% |
  | Methods (design 278, Data 506, Gold standards 928, formulae 296, Correlations 363, naming 310) | 2,681 | 31% |
  | Results | 2,179 | 25% |
  | Discussion — *Limitations* 276, *Comparison with prior research* 74, *Answers to research questions* 1,607 | 1,957 | 22% |
  | Conclusions | 323 | 4% |
- **First two paragraphs:** brutally fast. The entire Introduction before the first subsection is **49 words**: "Citation-based indicators are widely used to support research evaluations of individuals, departments, universities and countries (De Bellis, 2009; Moed, 2006). This article investigates two separate issues with the same data: whether OpenAlex is a suitable database for citation analysis, and which is the best citation-based indicator." One sentence of context, one sentence of contribution. The literature review then sits *inside* the Introduction as two subsections, and the Introduction ends with six numbered research questions, each with a one-sentence rationale ("RQ1: Are OpenAlex citation counts better… The apparently greater coverage of OpenAlex may make its citation counts more useful, although if it covers low quality articles then their citations may degrade the overall value").
- **Related work:** ~1,000 words, folded into the Introduction rather than given its own top-level section. Only ~40 references total.
- **Validation vs findings:** Methods (31%) is mostly *construction of the gold standard and the comparison design* — "Gold standards" alone is 928 words and openly explains that the primary gold standard is ChatGPT scores and the secondary is departmental REF averages. Validation argumentation is not a separate section: it is distributed between Methods (why this benchmark) and Limitations (why it may not hold). Findings + interpretation (Results 2,179 + Answers to RQs 1,607 = 3,786) are **1.4× the Methods** and ~9× the Limitations.
- **Limitations:** a named subsection and the **very first thing in the Discussion**, 276 words, and it is allowed to weaken the paper's own claim: "It is therefore unsafe to draw strong conclusions about the relative strengths of citation counts, NCS and NLCS."
- **Discussion structure worth copying:** *Limitations* → *Comparison with prior research* (74 words) → *Answers to research questions* (1,607), which walks RQ1…RQ6 in order. The reader can check that every question asked in the Introduction is answered.
- **References:** ~40 entries (OpenAlex matches 28). **Tables:** 4. **Figures:** 12 (one per field/UoA panel) — figures carry the field-level heterogeneity that prose could not.

## Paper 4 — Wu, Zhang & Zhao (2025), JASIST — read in full

- **Title:** Automated novelty evaluation of academic paper: A collaborative approach integrating human expertise and large language models
- **Year / venue / DOI:** 2025, JASIST, 10.1002/asi.70005; accepted manuscript read at https://arxiv.org/abs/2507.11330
- **Abstract:** 239 words, unstructured, problem-first ("Novelty is a crucial criterion in the peer review process… Both methods have limitations: experts have limited knowledge, and the effectiveness of the combination method is uncertain"), then approach, then a bare performance claim ("Extensive experiments demonstrate that our method achieves superior performance") — weaker than Thelwall's numeric abstract.
- **Section structure with measured lengths:**
  | Section | Words |
  |---|---|
  | 1. Introduction | 2,347 |
  | 2. Related Work | 1,753 |
  | 3. Dataset | 1,261 |
  | 4. Methodology | 810 |
  | 5. Experiments (incl. ablations, baselines) | 3,400 |
  | 6. Discussion | 1,477 |
  | 7. Conclusion and Future works | ~380 |
  | References | 2,940 (~88 entries) |
  | 9. Appendix (prompt examples, score definitions) | 1,410 |
- **First two paragraphs:** ¶1 defines the construct and cites the canonical definition ("novelty is defined as the reorganization of existing knowledge in an unprecedented manner (Schumpeter, 1939; Nelson & Winter, 1982)"), then immediately lists what is wrong with the standard measure ("These methods have certain drawbacks. Firstly, the extent to which cited publications serve as sources of inspiration for a paper is not yet well understood"). ¶2 continues the enumerated critique (citation bias, citation intent) and pivots to the opportunity (open peer review text + LLMs). Contribution is stated later in the Introduction as an explicit contributions list — this Introduction is long (2,347 words, ~15% of the paper) because it doubles as the motivation *and* the contribution statement.
- **Related work:** its own section, 1,753 words, immediately after the Introduction.
- **Validation vs findings:** "Experiments" (3,400) is the largest section and contains baselines, ablations and error analysis — i.e., in an ML-flavoured JASIST paper the validation *is* the finding, but it is presented as comparative results against ~10 baselines, not as a description of the validation procedure. Dataset construction (1,261) is separated from Methodology (810) so that the resource and the model are auditable independently.
- **Limitations:** not a named section; handled as hedges in the Discussion ("our aim is not to replace human judgment but to assist in secondary evaluations") and as "Future works" in the Conclusion ("our current dataset is limited to the field of computer science and conference papers"). This is the weakest of the four on limitations and I would not copy it.
- **Appendix:** prompts, extra ChatGPT feedback examples, and the score rubric (Table 5) — everything a replicator needs but a reader does not.
- **References:** ~88. **Tables:** 5. **Figures:** 5 in main text + supplementary figures S6–S8.

### Genre contrast (read in full, for calibration): Liu & Wang (2025), "Red alert: Millions of 'homeless' publications in Scopus should be resettled", JASIST 10.1002/asi.25011, arXiv 2508.18146
JASIST also publishes **brief communications**: 4,919 words, 11 pages, 192-word abstract, 27 references, 1 table, 3 figures, sections *1. Introduction / 2. Data and methods / 3. Affiliated but country-undefined publications in Scopus / 4–5. Discussion / 6. Suggestions*. Methods is 900 words and literally prints the two database queries. If our paper's real contribution is one crisp measurement plus a call to action, this is a legitimate JASIST shape — but then the validation has to shrink too, not just the results.

---

## Lessons

Baseline for the implications below: our draft currently spends **~1,500 words on validation, ~675 on results, and cites 8 works** (validation:results ≈ 2.2:1).

1. **Put the whole claim in the first 50–150 words, before any literature.**
   Thelwall's entire pre-subsection Introduction is two sentences: "Citation-based indicators are widely used to support research evaluations… This article investigates two separate issues with the same data: whether OpenAlex is a suitable database for citation analysis, and which is the best citation-based indicator." Schindler ¶1 is four citation-free sentences of stakes.
   *For us:* draft two opening paragraphs — stakes, then "this article does X and finds Y" with a number in it — and forbid citations in ¶1. If our contribution currently surfaces only after the validation section, it is ~2,000 words too late.

2. **Convert the validation section into a methods section plus a limitations subsection, and let the findings section be the biggest of the two.**
   Schindler splits the two explicitly: *Results: Information Extraction Performance* (1,670 words) versus *Results: Analysis of Software Mentions* (4,241) — validation is **0.39×** the findings. Thelwall's Results + RQ answers (3,786) dwarf his 276-word Limitations.
   *For us:* our 2.2:1 ratio is inverted relative to every exemplar. Target roughly 1:2. Concretely: cut validation prose to ~600–700 words of "here is the accuracy and here is what it costs us", push the procedural detail to an appendix, and grow results to ~1,500–2,000 words. Nothing else on this list matters as much.

3. **End the Introduction with explicit, numbered research questions (or a bulleted contributions list) and then answer them in that order.**
   Thelwall lists RQ1–RQ6, each with a one-line rationale, and closes with a subsection literally titled "Answers to research questions" (1,607 words) that walks them in order. Schindler uses the bulleted equivalent: "In summary, our contributions include: *A large-scale analysis of software usage* across 3,215,386 scholarly publications…".
   *For us:* 675 words of results is what happens when there is no list of questions forcing one answer per question. Three or four RQs will generate the missing 1,000 words of results by construction, and reviewers can then verify the paper delivers what it promised.

4. **Give related work 1,000–2,000 words and 30–60 references; 8 is below the floor for this venue.**
   Measured: Schindler 1,993 words / 55 refs; Thelwall ~1,000 words inside the Introduction / ~40 refs; Wu 1,753 words / ~88 refs; even the 4,900-word brief communication has 27. Softcite has ~60–70.
   *For us:* 8 references will read as "the authors do not know the literature" regardless of the quality of the measurement. Budget ~30 minimum, and structure the review the way Schindler does — one subsection deriving the *requirements* our design must satisfy, one reviewing *what prior studies found* — so the citations do work rather than decorate.

5. **Name a Limitations subsection, put it early in the Discussion, and let it actually cost you something.**
   Schindler: "the F = 0.94 performance for RE might overestimate the true performance as it relies on results of F = 0.885 entity recognition." Thelwall opens his Discussion with Limitations and concedes "It is therefore unsafe to draw strong conclusions about the relative strengths of citation counts, NCS and NLCS." Both are ~280–540 words.
   *For us:* much of our 1,500 validation words is probably defensive hedging. Move the hedges into a 300–500-word Limitations subsection, lead with the sampling-frame bias (Schindler leads with "the selection of PMC as primary data source implies a bias towards Medicine"), and delete the hedging everywhere else.

6. **Report the artifact's scale in the abstract, with digits.**
   Schindler: "11.8 M software mentions… a knowledge graph consisting of more than 300 M triples"; Thelwall: "28.6 million articles… 8,704 correlations… 97,816 UK REF 2021 articles"; Softcite: "manual annotation of 4,971 academic PDFs". Abstract lengths cluster at 190–240 words (Softcite's 104 is the outlier and it is a pure resource paper).
   *For us:* write a 200–240-word abstract in which at least three numbers appear, and state the headline finding rather than "we validate our approach".

7. **Carry heterogeneity in figures, not prose, and push apparatus into an appendix.**
   Thelwall: 12 figures, 4 tables, for a 8,700-word paper — the per-field panels are the argument. Schindler: 15 figures, 14 tables, but 11 of the tables are appendix tables (hyper-parameter grids, domain categorisation), with one supplemental file. Wu's appendix holds the prompts and the scoring rubric.
   *For us:* every validation table that exists to prove diligence rather than to change a reader's belief belongs in supplementary material. That is also the cheapest way to hit lesson 2's word budget without deleting work.

8. **Write a dedicated software-and-data release paragraph with versions and DOIs.**
   Schindler's Conclusion contains a *Software and Data* subsection naming every dependency with version ("Python 3.9.5… PyTorch 1.9.0… R 4.1.1"), the analysis repo, and the Zenodo DOI; Softcite's contribution *is* the Zenodo/GitHub release plus its provenance description, which the abstract advertises as "an extended discussion of its creation process".
   *For us:* one named subsection near the end, with repo URL, archived DOI, licence, and versions. If the dataset is part of the contribution, say so in the title/abstract the way Softcite does ("Softcite dataset: A dataset of…") — that is also the only legitimate way to make validation-heavy proportions acceptable.

9. **Choose the genre deliberately: full research article or brief communication.**
   Liu & Wang's brief communication runs 4,919 words, 900 of them methods, 27 references, and ends with a *Suggestions* section aimed at named stakeholders. A full research article (Schindler 19,034; Wu ~13,000; Thelwall 8,738) carries a Related Work section and 40+ references.
   *For us:* at ~2,200 words of core content, the current draft is neither. Either grow it to ~7,000–9,000 words on the Thelwall template (the cheapest route: lessons 2–4), or reframe it as a brief communication — in which case the validation must come down to ~500 words and the paper must end with concrete recommendations to identifiable actors.

10. **Add a one-paragraph roadmap at the end of the Introduction.**
    Schindler: "The remaining paper is organized as follows. Related work is discussed in the following section, whereas the *Methods and Materials* introduces developed information extraction methods…". All four exemplars signpost; two do it in an explicit paragraph.
    *For us:* trivial to add, and it forces us to notice that the current outline has a 1,500-word section no reader would predict from the title.

---

### What this rests on
- Section word counts for Schindler are exact (JATS XML). Thelwall and Wu counts come from `pdftotext -layout` on the author PDFs, so they are accurate to a few percent (headers/footnotes/figure text leak in).
- Thelwall and Wu are **accepted manuscripts, not the Wiley-typeset versions**; Wiley blocks automated access, so heading names could have shifted in copy-editing. Section proportions will not have.
- Softcite's structure is **inferred**, not read. If the exact published section list matters for our restructuring, it needs a library copy of the PDF — everything else in this file is first-hand.
