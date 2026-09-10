# 10-page version

`paper10-overleaf.zip` in the repository root. Upload it to Overleaf with
**New Project → Upload Project**, then check **Menu → Compiler** says **pdfLaTeX**.

Verified from a clean unpack: 0 errors, 0 undefined references, 0 undefined citations,
0 overfull boxes.

```
main body      10 pages   sections 1 to 8, ending with Contribution and Acknowledgments
appendices     17 pages   A worked examples, B design and preregistration,
                          C atomic skills and integrity, D full limitations, E second series
references      2 pages   placed after the appendices
Russian block   1 page    title, abstract and keywords, as the journal template requires
```

## How the venue's requirements are met

| requirement | where |
|---|---|
| no longer than 10 pages | main body ends on page 10; everything after it is appendix |
| unlimited appendices | five appendices, pages 11 to 27 |
| not anonymized | real author block on page 1, no anonymity language anywhere |
| full list of authors | `\author` on page 1 |
| mentor listed | Acknowledgments, page 10 |
| Contribution section | Section 8, page 10 |

References sit **after** the appendices on purpose, so that the ten-page count is
unambiguous under either reading of the rule. If the venue counts references inside
the limit, move the `\providecommand{\bysame}` block back to just before
`\clearpage\appendix`; the body then runs to about 11.5 pages and needs roughly one
more page of cuts.

## What you must fill in

Two places, and they must agree with each other:

1. The author block at the top of `main10.tex`, currently `Author One` to
   `Author Four` with one affiliation and one correspondence address.
2. Section 8, `Contribution`, currently four paragraphs keyed to the same four
   placeholder names.

The four contribution paragraphs are not invented. They are the work split the
project specified for itself in
`verifier_bottleneck_codex_pack/docs/11_four_person_work_split.md`: the environment
and verifier, the exploration study, the low-rank and on-policy training, and the
experimental design and statistics. Assign the real names and adjust each paragraph
to what that person actually did. The repository history shows two contributors, so
if the team is smaller than four, merge the paragraphs rather than leaving a
placeholder standing.

The mentor is named as **Serguei Barannikov** in the Acknowledgments, which follows
`SERGUEI_PROTOCOL.md` in this repository. Confirm the spelling and the affiliation.

`\originfo{1}{}{}{2026}` still carries a placeholder volume number.

## Relationship to the full version

`paper/` holds the 44-page version. This one is not a rewrite: the prose was taken
from it and from the Russian source document and then shortened, and every number
was re-checked against the same statistics files. Nothing that carries a claim was
deleted; detail was moved into the appendices. The two versions can be maintained
together, because both `\input` the same `tab_*.tex` fragments and use the same
figures.
