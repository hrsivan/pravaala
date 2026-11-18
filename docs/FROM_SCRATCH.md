# Vidruma from scratch

This document builds the project up in the order in which one would build it:
**define the task → decide how success is measured → prepare the data →
understand the data → train → evaluate → analyse.** A tiny toy corpus runs
through every step, small enough to compute by hand. Every toy number below is
printed by

```bash
python docs/toy_example.py
```

so you can check your hand calculations against it. Boxes marked
**At scale** say what the same step looks like on the real 5.8M-word corpus.
Prerequisites: high-school probability (what $P(A)$, $P(A,B)$ and $P(A\mid B)$
mean) and logarithms. Everything else is built here.

Contents
1. [The task](#1-the-task)
2. [Evaluation first](#2-evaluation-first-how-we-will-know-the-vectors-are-good)
3. [Data: from text to tokens](#3-data-from-text-to-tokens)
4. [Measuring the data: entropy and conditional entropy](#4-measuring-the-data-entropy-and-conditional-entropy)
5. [Training: from counts to vectors (PMI in detail)](#5-training-from-counts-to-vectors)
6. [Running the evaluation on the toy](#6-running-the-evaluation-on-the-toy)
7. [Back to the real system](#7-back-to-the-real-system)
8. [Notation](#8-notation)

---

## 1. The task

**Input.** A large collection of Sanskrit text (the *corpus*).

**Output.** For every word $w$ in a vocabulary $\mathcal{V}$, a vector
$\mathbf{v}(w) \in \mathbb{R}^d$ (typically $d = 100$ or $300$). Formally we
learn a function $\mathbf{v} : \mathcal{V} \to \mathbb{R}^d$.

**What should the vectors do?** Words with similar meanings should get
vectors pointing in similar directions. Similarity of direction is measured by
the cosine of the angle between the vectors:

$$\cos(\mathbf{u},\mathbf{v}) = \frac{\mathbf{u}\cdot\mathbf{v}}{\|\mathbf{u}\|\,\|\mathbf{v}\|} \in [-1, 1].$$

**How can text alone tell us about meaning?** Through the *distributional
hypothesis* (Harris 1954; Firth 1957): words that occur in similar contexts
tend to have similar meanings. *jala* and *toya* (both "water") are both
drunk, both found in rivers. We never tell the model what words mean; it
infers similarity from shared company.

**What is special about Sanskrit.** A "word" in running Sanskrit text is not
an obvious unit. Sounds merge across word boundaries (*sandhi*), words fuse
into compounds, every noun has up to 24 inflected forms, and one spelling can
belong to two different words (*homographs*). Deciding *what counts as a word*
before learning vectors is therefore half the problem. Our research questions
are about exactly this:

* **RQ1** Do vectors improve if we first reduce words to their dictionary
  form (lemma), or to a disambiguated lemma ID?
* **RQ2** Do the vectors recover the Amarakośa's grouping of words into
  semantic chapters (*vargas*)?
* **RQ3** How much of the "missing word" problem comes from inflection, and
  how much ambiguity does lemmatisation resolve?

---

## 2. Evaluation first: how we will know the vectors are good

In any NLP project, define the evaluation *before* training. Otherwise you
risk choosing the evaluation that flatters the model.

### 2.1 The gold standard: the Amarakośa

The Amarakośa is a classical thesaurus. Its structure gives two levels of
"meaning group":

* **synset** — words that are synonyms (*jala*, *toya*, *salila*, … "water");
* **varga** — a chapter grouping many synsets by topic (the *vāri-varga*,
  "water chapter", holds water, rivers, fish, boats, …).

**Toy gold.** Two vargas, four synsets:

| varga | synsets |
|---|---|
| water | {jala, toya, salila} "water", {nadī} "river" |
| fire | {agni, vahni} "fire", {dīpa} "lamp" |

### 2.2 Three tasks built from the gold

1. **Relatedness.** Pairs of words with a label $y$: 2 = same synset,
   1 = same varga only, 0 = unrelated. A good model gives higher cosines to
   higher labels.
   *Toy:* (jala, toya, 2), (agni, vahni, 2), (jala, nadī, 1),
   (agni, dīpa, 1), (jala, agni, 0), (toya, dīpa, 0), (salila, jala, 2).
2. **Synonym multiple choice (MCQ).** Given a word and four options, pick the
   synonym. *Toy:* *jala* → {toya ✓, nadī, agni, dīpa}. The model picks the
   option with the highest cosine.
3. **Clustering.** Cluster the vectors into $k$ groups and compare the groups
   with the vargas. *Toy:* six words, $k = 2$.

### 2.3 Metrics

* **Coverage.** If a word has no vector (it never occurred in training, or
  too rarely), it is *out of vocabulary* (OOV). We drop items containing OOV
  words and report the fraction kept. *Toy:* *salila* never occurs, so
  coverage is 6/7.
* **Spearman ρ.** Rank all pairs by cosine and by label, then correlate the
  ranks. +1 means the cosine ordering matches the label ordering perfectly.
  Only ranks matter, so a model whose cosines are all squeezed into
  [0.7, 0.9] is not penalised for its scale.
* **AUC.** Take one random pair with label 2 and one with label 0. AUC is the
  probability that the label-2 pair has the higher cosine (ties count ½).
  0.5 is chance, 1.0 is perfect. AUC(1 vs 0) asks the same about same-varga
  vs unrelated pairs.
* **Accuracy** for the MCQ (chance = 25%).
* **Purity, NMI, VI** for clustering; computed by hand in §6.

### 2.4 Rules that keep the evaluation honest

* **Dev vs test.** Anything we *choose* (thresholds, dimension, window size)
  is chosen on a development split. The test split is touched once.
* **No imputation.** EvalSan gives OOV words the average vector, so two OOV
  words get cosine 1 and look like synonyms. We exclude them instead.
* **No leakage.** The model must not see the answers during training. **At
  scale** this is a real issue: DCS *contains the Amarakośa itself* and eight
  other synonym lexicons (13,095 sentences). A model trained on all of DCS
  has read the test dictionary's synonym lists as running text. We therefore
  also train on DCS minus these lexicons and compare (§7).

---

## 3. Data: from text to tokens

### 3.1 Transliteration

Sanskrit can be written in many ASCII schemes. In **SLP1** every sound is
exactly one letter. In **WX** also, but with *different* letters: SLP1 `t` is
dental *t*, WX `t` is retroflex *ṭ*. *amṛta* is `amfta` in SLP1 and `amqwa` in
WX. If the corpus is in one scheme and the benchmark in the other, lookups
fail silently. **At scale** this is what happened in EvalSan: its relatedness
and synonym files are WX while its vectors are SLP1. We convert everything to
SLP1 (`src/vidruma/translit.py`).

### 3.2 The toy corpus in three views

Ten sentences. Each is shown as written (**surf**), with sandhi undone
(**seg**), and reduced to lemmas (**lem**):

| # | surf (as written) | seg (sandhi undone) | lem (dictionary forms) | meaning |
|---|---|---|---|---|
| 1 | jalaM pibati | jalam pibati | jala pA | (he) drinks water |
| 2 | toyaM pibati | toyam pibati | toya pA | (he) drinks water |
| 3 | nadyAM jalam | nadyAm jalam | nadI jala | water in the river |
| 4 | nadyAM toyam | nadyAm toyam | nadI toya | water in the river |
| 5 | agnirdahati | agniH dahati | agni dah | fire burns |
| 6 | vahnirdahati | vahniH dahati | vahni dah | fire burns |
| 7 | dIpe'gniH | dIpe agniH | dIpa agni | fire in the lamp |
| 8 | dIpe vahniH | dIpe vahniH | dIpa vahni | fire in the lamp |
| 9 | jalamagniSca | jalam agniH ca | jala agni ca | water and fire |
| 10 | toyaM vahniSca | toyam vahniH ca | toya vahni ca | water and fire |

Look at sentence 7: written *dīpe 'gniḥ*, the *a* of *agniḥ* has merged into
the preceding *e*. A model reading `dIpe'gniH` as one token never sees *agni*
there at all. Sentence 9 fuses three words into one string.

### 3.3 Types, tokens, sparsity

A *token* is an occurrence; a *type* is a distinct string.

| view | tokens | types |
|---|---|---|
| surf | 16 | 13 |
| seg | 22 | 9 |
| lem | 22 | 9 |

Sandhi hides tokens (16 instead of 22) and multiplies types. In the toy, seg
and lem happen to have the same number of types; at scale, inflection
multiplies them heavily (306,810 seg types vs 90,856 lemma types). The more
types, the fewer occurrences per type, and the less evidence per vector. This
is *sparsity*. A model-free measure of it is the **Good–Turing estimate**: if
$N_1$ types occurred exactly once among $N$ tokens, the probability that the
next token is a never-seen type is about $N_1/N$. **At scale:** 0.162 (surf),
0.024 (seg), 0.0065 (lem).

---

## 4. Measuring the data: entropy and conditional entropy

Lemmatising throws information away (case, number, tense) and resolves some
ambiguity. Information theory lets us measure both in bits. This section
builds the tools from nothing.

### 4.1 Surprise

If an event has probability $p$, its **surprisal** (self-information) is

$$I = -\log_2 p \quad \text{bits}.$$

A fair coin landing heads: $p = 1/2$, $I = 1$ bit. A specific face of a fair
die: $I = \log_2 6 \approx 2.58$ bits. A certain event: $I = 0$. Rarer events
are more surprising, and surprises of independent events add up
($-\log(pq) = -\log p - \log q$). That additivity is why we use the log.

### 4.2 Entropy: average surprise

For a random variable $X$ taking values $x$ with probabilities $p(x)$:

$$H(X) = \sum_x p(x)\,(-\log_2 p(x)).$$

It is the average number of bits needed to identify the outcome, or the
minimum average number of yes/no questions in an optimal guessing game. A
fair coin: 1 bit. A fair die: 2.58 bits. A coin that always lands heads: 0
bits.

**Estimating it from data.** We don't know $p$, so we use relative frequencies
$\hat p(x) = \#(x)/N$ and plug them in (the *plug-in estimator*). With small
samples this slightly underestimates the true entropy; at our corpus sizes
the bias is negligible.

### 4.3 Joint and conditional entropy

Now observe two things at once about each word occurrence: its inflected form
$S$ (seg) and its lemma $L$ (lem).

* **Joint entropy** $H(S, L)$: the entropy of the *pair*, computed exactly
  like $H$ but over pairs $(s, \ell)$.
* **Conditional entropy** $H(L \mid S)$: the average remaining uncertainty
  about the lemma *after* you have been told the form. Formally,

$$H(L \mid S) = \sum_s p(s)\, H(L \mid S = s), \qquad H(L \mid S = s) = -\sum_\ell p(\ell \mid s)\log_2 p(\ell \mid s).$$

  In words: for each form, how unsure are you about its lemma; then average,
  weighting each form by how often it occurs.

* **Chain rule**: $H(S, L) = H(S) + H(L \mid S)$. "Identify both" = "identify
  the form" + "then identify the lemma given the form". This gives the
  formula the code uses:

$$H(L \mid S) = H(S, L) - H(S).$$

### 4.4 A hand-computable example

Ten occurrences:

| form (seg) | lemma | count | note |
|---|---|---|---|
| bhavati | bhū | 3 | "(he) becomes" |
| bhavati | bhavat | 1 | "in you", locative of *bhavat* |
| jalam | jala | 4 | |
| jalena | jala | 2 | "by water" |

**$H(\text{lem} \mid \text{seg})$, by the definition.**
* Given *bhavati* (4 of 10 occurrences), the lemma is *bhū* with probability
  3/4 and *bhavat* with 1/4:
  $H = -\tfrac34\log_2\tfrac34 - \tfrac14\log_2\tfrac14 = 0.811$ bits.
* Given *jalam* or *jalena*, the lemma is certainly *jala*: 0 bits.
* Average: $0.4 \times 0.811 + 0.6 \times 0 = 0.325$ bits.

**The same, by the chain rule.** $H(\text{seg}) = 1.522$ (forms 4, 4, 2 out
of 10); $H(\text{seg}, \text{lem}) = 1.846$ (pairs 3, 1, 4, 2);
$1.846 - 1.522 = 0.325$ ✓.

**The other direction, $H(\text{seg} \mid \text{lem})$.** Given lemma *jala*
(6 of 10), the form is *jalam* (4/6) or *jalena* (2/6): 0.918 bits. Given
*bhū* or *bhavat*, the form is certain. Average: $0.6 \times 0.918 = 0.551$
bits.

**Reading the two directions.**
* $H(\text{lem} \mid \text{seg})$ is the **ambiguity** a lemmatiser must
  resolve: how often one written form could be two different words.
* $H(\text{seg} \mid \text{lem})$ is the **information lemmatisation throws
  away**: the grammatical detail (case, number, …) that distinguishes forms of
  one lemma.

They are not equal, because ambiguity and inflection are different phenomena.

**Lemma IDs.** Same idea one level down: if *brahman* occurs 3 times as the
neuter lemma (the Absolute) and once as the masculine (Brahmā), and *deva*
4 times with one ID, then $H(\text{lid} \mid \text{lem}) = 0.5 \times 0.811 = 0.406$
bits. That is the homography the ID resolves.

### 4.5 How the code computes it on DCS

1. `scripts/build_corpora.py` writes one file per view, one sentence per line,
   **aligned**: the $i$-th token of a line in `seg.txt` is the same word
   occurrence as the $i$-th token of that line in `lem.txt`.
2. `scripts/oov_report.py` reads the files in parallel. Wherever the lines
   have equal length, it zips them into pairs (form, lemma), one per
   occurrence: 5.8 million pairs.
3. `infotheory.conditional_entropy(xs, ys)` counts the pairs and the $x$
   values, and returns $H(X, Y) - H(X)$ from those counts.

**At scale** (bits per token):

| quantity | value | meaning |
|---|---|---|
| $H(\text{lem} \mid \text{seg})$ | 0.100 | inflected forms are rarely ambiguous between lemmas |
| $H(\text{lid} \mid \text{lem})$ | 0.115 | lemma IDs resolve little homography |
| $H(\text{seg} \mid \text{lem})$ | 2.40 | lemmatisation discards a lot of grammar |
| $H(\text{sense} \mid \text{lid})$ | 0.94 | (on sense-annotated text) most ambiguity is *polysemy within* one lemma ID |

A useful way to feel these numbers: $2^{H}$ is the *effective number of
equally likely choices*. $2^{0.115} \approx 1.08$: after knowing the lemma
string, you face on average about 1.08 equally likely IDs. $2^{2.40} \approx
5.3$: a lemma shows up as about five "effective" forms. $2^{0.94} \approx 1.9$:
about two effective senses per lemma ID. This is why lemma IDs alone cannot
deliver sense-level vectors.

---

## 5. Training: from counts to vectors

### 5.1 Contexts and the co-occurrence matrix

Choose what counts as "company". In the toy we use the whole sentence: two
words co-occur if they are in the same sentence. (At scale we use a window of
±5 words, with nearer words counting more.) Count, for every pair of words
$(w, c)$, how often $c$ appears in the context of $w$:

|      | jala | toya | nadI | pA | agni | vahni | dah | dIpa | ca |
|---|---|---|---|---|---|---|---|---|---|
| **jala** | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 1 |
| **toya** | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 0 | 1 |
| **nadI** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **pA** | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **agni** | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| **vahni** | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| **dah** | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| **dIpa** | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |
| **ca** | 1 | 1 | 0 | 0 | 1 | 1 | 0 | 0 | 0 |

Total $N = 28$. Row sums: 4 for jala, toya, agni, vahni and ca; 2 for the
others. Each row is already a vector: *jala*'s row says "jala keeps company
with nadī, pā, agni, ca".

### 5.2 Why raw counts are not enough

Cosine on raw rows: cos(jala, toya) = 0.750, cos(jala, agni) = 0.250. The
water–fire similarity is inflated by the shared context *ca* ("and"), which
says nothing about meaning. In real text, particles like *ca*, *iti*, *api*
co-occur with everything and dominate raw counts. We need a measure that asks
whether $c$ appears near $w$ **more than we would expect by chance**.

### 5.3 Pointwise mutual information, from the ground up

**Step 1: probabilities from counts.** Pick a random cell occurrence of the
co-occurrence table. Then

$$P(w, c) = \frac{C_{wc}}{N}, \qquad P(w) = \frac{\sum_c C_{wc}}{N}, \qquad P(c) = \frac{\sum_w C_{wc}}{N}.$$

$P(w)$ and $P(c)$ are the *marginals*: how often $w$ (or $c$) is involved at
all. *Toy:* $P(\text{jala}, \text{pA}) = 1/28$, $P(\text{jala}) = 4/28$,
$P(\text{pA}) = 2/28$.

**Step 2: what "by chance" means.** If $w$ and $c$ were *independent*, i.e.
knowing one tells you nothing about the other, the probability of seeing them
together would be the product $P(w)\,P(c)$. Independence is the "no
relationship" baseline.

**Step 3: compare observed with expected.**

$$\frac{P(w,c)}{P(w)\,P(c)} \quad \begin{cases} > 1 & \text{together more often than chance (attraction)}\\ = 1 & \text{exactly chance (independence)}\\ < 1 & \text{less than chance (avoidance)}\end{cases}$$

**Step 4: take the log.**

$$\mathrm{PMI}(w,c) = \log_2 \frac{P(w,c)}{P(w)\,P(c)}.$$

Now independence is 0, attraction is positive and avoidance negative, and
the scale is bits.

**Toy, by hand.**
* $\mathrm{PMI}(\text{jala}, \text{pA}) = \log_2 \frac{1/28}{(4/28)(2/28)} = \log_2 3.5 = 1.807$ bits.
* $\mathrm{PMI}(\text{jala}, \text{ca}) = \log_2 \frac{1/28}{(4/28)(4/28)} = \log_2 1.75 = 0.807$ bits.

Both pairs occur once, but *ca* is involved in twice as many co-occurrences
as *pā*. Its company with *jala* is therefore less surprising, and PMI
discounts it by exactly one bit.

**Three equivalent readings** (all follow from the definition):

1. *Change in surprise.* $\mathrm{PMI}(w,c) = \log_2 \frac{P(c \mid w)}{P(c)} = I(c) - I(c \mid w)$:
   how many bits of surprise about $c$ are removed by knowing that $w$ is
   nearby. For (jala, pā): $I(\text{pA}) = \log_2 14 = 3.807$ bits;
   $I(\text{pA} \mid \text{jala}) = \log_2 4 = 2$ bits; difference 1.807 ✓.
2. *Surprisal bookkeeping.* $\mathrm{PMI} = I(w) + I(c) - I(w, c)$: the bits
   you would need for $w$ and $c$ separately, minus the bits for the pair.
   If the pair is cheaper to describe than its parts, the words "explain"
   each other.
3. *Symmetry.* $\mathrm{PMI}(w,c) = \mathrm{PMI}(c,w)$; the relationship has
   no direction.

**From pointwise to average: mutual information and KL.** Average PMI over
all pairs, weighted by how often they occur, and you get the **mutual
information** $I(W; C) = \sum_{w,c} P(w,c)\,\mathrm{PMI}(w,c)$, the average
information that knowing a word gives about its context. Average over just
*one* word's contexts and you get

$$\sum_c P(c \mid w)\,\mathrm{PMI}(w,c) = \sum_c P(c \mid w)\log_2\frac{P(c \mid w)}{P(c)} = D_{\mathrm{KL}}\big(P(\cdot \mid w)\,\big\|\,P(\cdot)\big).$$

This is the **Kullback–Leibler divergence**: how different $w$'s company is
from the company of a random word. *Toy:* KL(jala) = 1.307 bits, KL(ca) =
0.807 bits. *ca* keeps the most "average" company and is the least
informative. This is the precise sense in which PMI reduces the effect of
frequent, unselective words: they are discounted for being *unselective*, not
merely for being frequent.

**Why the positive part (PPMI).** Most word pairs never co-occur, so their
estimated $P(w,c) = 0$ and $\mathrm{PMI} = -\infty$. That reflects too little
data, not true avoidance. Negative estimates are noisy in general, so we
keep only the positive part:
$\mathrm{PPMI}(w,c) = \max(\mathrm{PMI}(w,c), 0)$.

**PMI's weakness: rare events.** If $w$ and $c$ each occur exactly once, and
together, then $\mathrm{PMI} = \log_2 \frac{1/N}{(1/N)(1/N)} = \log_2 N$, the
maximum possible value. One coincidence produces the strongest association
in the table. Remedies used in our code:
* a minimum count (words seen fewer than 5 times get no vector);
* **context smoothing**, which replaces $P(c)$ by
  $P_\alpha(c) \propto \#(c)^{0.75}$. Raising counts to a power below 1
  inflates rare contexts' probabilities a little, which shrinks their PMIs;
* averaging over many contexts (via SVD, next).

*What about language-model smoothing?* We tested the two methods that address
rarity of the *pair* rather than of the word or context
(`scripts/sweep_smoothing.py`, `results/all_nolex+np/sweep_smoothing_lem@clean.json`):
absolute discounting, $\mathrm{PMI}_D = \mathrm{PMI} + \log(1 - D/r)$ with
$r$ the pair's count and Ney's $D = n_1/(n_1+2n_2) \approx 0.66$; and the
Kneser–Ney continuation marginal, $Q(c) \propto$ number of distinct partners
of $c$. Neither helps. The diagnostics (`scripts/smoothing_diagnostics.py`)
say why:
* single co-occurrences are 62% of all cells and 64% of the top 1% of PPMI
  cells. After the minimum count and context smoothing they are no longer
  over-represented at the top, so there is little left to correct;
* for words seen 5–20 times, 76% of the PPMI row mass comes from single
  co-occurrences. Discounting removes most of the evidence for exactly the
  rare words the benchmark needs;
* distinct partners grow as $\#(c)^{0.82}$, so the continuation marginal is
  context smoothing with $\alpha \approx 0.82$, nearly the $0.75$ we already use.

Lowering the minimum count from 5 to 2, by contrast, raises coverage
(relatedness 48.5 → 57.7%, synonym items 19.0 → 27.9%) without changing
quality on the items the baseline already covered.

Code: `infotheory.ppmi_matrix` implements PMI with smoothing $\alpha$, an
optional shift (§5.5), absolute discounting (`R=`, `discount=`) and the
continuation marginal (`ctx="cont"`).

**Toy PPMI rows** (bits; columns in the same order as the table above):

```
jala  [0     0     1.807 1.807 0.807 0     0     0     0.807]
toya  [0     0     1.807 1.807 0     0.807 0     0     0.807]
agni  [0.807 0     0     0     0     0     1.807 1.807 0.807]
ca    [0.807 0.807 0     0     0.807 0.807 0     0     0    ]
```

Cosine on PPMI rows: cos(jala, toya) = **0.917** (up from 0.750) and
cos(jala, agni) = **0.083** (down from 0.250). PMI sharpened the contrast.

### 5.4 From long sparse rows to short dense vectors: SVD

Real PPMI rows have one entry per vocabulary word: tens of thousands of
numbers, mostly zero. The **singular value decomposition** factorises the
matrix as $M = U\Sigma V^\top$, where $\Sigma$ holds non-negative *singular
values* in decreasing order. Keeping only the top $d$ gives the best possible
rank-$d$ approximation of $M$ (Eckart–Young theorem). Rows of
$U_d\Sigma_d^{1/2}$ are the word vectors. Intuitively, SVD finds the $d$
directions along which the rows vary most, and describes each word by its
position along those directions. Words with similar rows land close
together, *even if they never share an exact context*.

**Toy lesson about $d$.** The toy PPMI matrix has four large singular values
(4.38, 4.04, 3.58, 3.23), one per natural group of contexts. With $d = 4$:
cos(jala, toya) = 1.000, cos(jala, agni) = 0.044. With $d = 2$ the space is
too small, and two *unrelated* groups are forced together: cos(jala, dah) =
0.999. Too few dimensions merge distinctions; too many keep noise. $d$ is
chosen on dev data (at scale, $d = 300$ won).

Code: `embed.svd_embeddings`.

### 5.5 Other ways of estimating the same thing

* **SGNS (word2vec).** Trains vectors by gradient descent so that real
  (word, context) pairs score high and random pairs score low. Levy & Goldberg
  (2014) proved that at the optimum $\mathbf{w}\cdot\mathbf{c} = \mathrm{PMI}(w,c) - \log k$,
  with $k$ random pairs per real one. So SGNS is another factorisation of
  (shifted) PMI.
* **GloVe.** Fits $\mathbf{w}\cdot\mathbf{c} + b_w + b_c \approx \log C_{wc}$
  by weighted least squares. The biases absorb $\log P(w)$ and $\log P(c)$,
  so again the dot product targets PMI.
* **FastText.** SGNS where a word's vector is the sum of its character-chunk
  vectors, so it can produce vectors for unseen words.

All four models therefore aim at the same information-theoretic object, and
differ in how they estimate it.

**Frequency damping in bits.** word2vec randomly drops occurrences of very
frequent words. The keep-probability $\sqrt{t/f(w)}$ can be rewritten as
$2^{-(I_t - I(w))/2}$, with $I(w) = -\log_2 f(w)$ the word's surprisal and
$I_t = -\log_2 t$. Words carrying fewer than $I_t$ bits are thinned out.
**At scale:** *ca* carries 5.0 bits, $I_t = 13.3$, so *ca* is kept 5.7% of
the time.

---

## 6. Running the evaluation on the toy

Vectors = PPMI rows (§5.3).

**Relatedness.**

| pair | gold | cosine |
|---|---|---|
| jala – toya | 2 | 0.917 |
| agni – vahni | 2 | 0.917 |
| jala – nadī | 1 | **0.000** |
| agni – dīpa | 1 | **0.000** |
| jala – agni | 0 | 0.083 |
| toya – dīpa | 0 | 0.204 |
| salila – jala | 2 | OOV, dropped |

* Coverage 6/7.
* AUC(2 vs 0) = **1.0**: every synonym pair beats every unrelated pair.
* AUC(1 vs 0) = **0.0**: every *same-varga* pair scores *below* every
  unrelated pair.
* Spearman ρ = 0.492.

**Why do same-varga pairs fail?** *jala* and *nadī* co-occur ("water in the
river"), but they are not used *in the same kinds of context*. Water is drunk;
rivers contain water. Their PPMI rows share no non-zero column, so the cosine
is 0. Linguists distinguish two kinds of relation:

* **syntagmatic** — words that occur *together* (jala–nadī, agni–dīpa);
* **paradigmatic** — words that occur *in the same slots*, i.e. can replace
  each other (jala–toya).

Cosine between context vectors measures *paradigmatic* similarity. A varga
mixes both kinds. That is why, **at scale**, the same-varga vs unrelated AUC
is only about 0.6 while synonym vs unrelated is about 0.88, and why the 3-way
relatedness F1 has a ceiling near 0.55.

**Synonym MCQ.** *jala* → toya 0.917, nadī 0.000, agni 0.083, dīpa 0.204 →
picks **toya** ✓.

**Clustering.** k-means with $k = 2$ on the six words gives
{jala, toya, dīpa} and {nadī, agni, vahni}. Compare with gold
{jala, toya, nadī | agni, vahni, dīpa}:

* **Purity**: label each cluster with its majority gold class and count the
  matches: 2 + 2 = 4 of 6 = **0.667**.
* **Entropies**: $H(K) = 1$ (clusters 3 + 3), $H(G) = 1$ (classes 3 + 3),
  $H(K, G) = 1.918$ (cells 2, 1, 1, 2 of 6).
* **Mutual information** $I(K; G) = H(K) + H(G) - H(K, G) = 0.082$ bits:
  knowing a word's cluster tells you almost nothing about its varga.
* **NMI** $= 2I/(H(K) + H(G)) = $ **0.082**, on a 0–1 scale.
* **VI** $= H(K \mid G) + H(G \mid K) = 2H(K, G) - H(K) - H(G) =$ **1.837**
  bits (0 would be a perfect match; the maximum here is $\log_2 6 = 2.58$).

Purity (0.667) looks respectable, but NMI (0.08) reveals the clustering is
nearly uninformative. With 2 clusters, purity can never fall below 0.5, so
purity alone flatters weak results. This is why we always report NMI or VI,
and compare against a random baseline (the *permutation null*: shuffle the
gold labels many times and see what scores chance produces).

---

## 7. Back to the real system

| step | toy | at scale |
|---|---|---|
| corpus | 10 sentences | DCS: 770,945 sentences, 5.8M words |
| views | surf / seg / lem | + lid (lemma IDs), sense |
| context | whole sentence | ±5 words, weight 1/distance |
| vectors | PPMI rows | PPMI-SVD, SGNS, GloVe, FastText; $d$ = 100–300 |
| relatedness | 7 pairs | 4,551 test pairs; best ρ ≈ 0.56, AUC(2v0) ≈ 0.89 |
| MCQ | 1 item | 3,034 items; best ≈ 71% on held-out items |
| clustering | 6 words, $k = 2$ | ~4,500 Amarakośa words, $k = 22$ vargas; NMI 0.16–0.25 vs null 0.02 |

What the real system adds on top of the toy:

* **OOV accounting.** Why each missing word is missing: encoding, sandhi,
  inflection, homograph splitting, or genuine rarity. Changing one step at a
  time and recording the OOV rate gives a telescoping sum.
  (`src/vidruma/oov.py`)
* **Per-varga analysis.** Which vargas are recovered, measured by *specific
  information*: how many bits knowing a word's varga tells you about its
  cluster, corrected for chance. (`scripts/analyse_vargas.py`)
* **Leak control.** Training with and without the lexicons inside DCS.
* **Protocols.** Our protocol (exclusion, rank statistics, dev-tuned
  thresholds, nulls) side by side with an EvalSan-faithful replication.

The step-by-step code map is in `docs/EXPLAINER.md`.

---

## 8. Notation

| symbol | read as |
|---|---|
| $P(w,c)$, $P(w)$ | joint and marginal probability (estimated by relative frequency) |
| $P(c \mid w)$ | probability of context $c$ given word $w$ $= P(w,c)/P(w)$ |
| $I(x) = -\log_2 p(x)$ | surprisal of outcome $x$ (bits) |
| $H(X)$ | entropy: average surprisal of $X$ |
| $H(X, Y)$, $H(Y \mid X)$ | joint entropy; conditional entropy $= H(X,Y) - H(X)$ |
| $I(X; Y)$ | mutual information $= H(X) + H(Y) - H(X,Y)$ |
| $D_{\mathrm{KL}}(p \,\|\, q)$ | KL divergence: how different $p$ is from $q$ |
| $\mathrm{PMI}(w,c)$, $\mathrm{PPMI}$ | pointwise mutual information; its positive part |
| $\cos(\mathbf{u}, \mathbf{v})$ | cosine of the angle between two vectors |
| AUC | probability that a random higher-label pair outscores a random lower-label pair |
| NMI, VI | normalised mutual information; variation of information |
