# Sparsity and temporal processing: the question, and what an answer needs

> *"Induce different levels of sparsity in the hidden layers (through
> regularisation), then observe whether sparser-activity networks do more
> temporal processing than denser ones."* — supervisor's suggestion

**Hypothesis (H1):** when the hidden-layer activity becomes more sparse, the network
uses the exact time of the hidden spikes more. It uses a rate code less.

This document is the **conceptual reference**. It tells you what the question means,
how to measure each side of the question, why an honest test is difficult, and what
counts as an answer. It contains no scripts, no grids, no file paths and no
hyper-parameters. Those are in the execution logs below. The logs also record what
happened when we did the tests.

**Read the documents in this order:**

| # | Document | What it is |
|---|---|---|
| 1 | **this file** | the question and the conceptual landscape |
| 2 | [sparse_network_test_progress.md](../legacy/sn_progress/sparse_network_test_progress.md) | v1 execution log — how to make a sparsity gradient exist |
| 3 | [sparse_network_1stLayer_results.md](../legacy/sn_progress/sparse_network_1stLayer_results.md) | v1 results — four perturbations, two arms, and the diagnosis |
| 4 | [sparse_network_test_progress_v2.md](../legacy/sn_progress/sparse_network_test_progress_v2.md) | v2 execution log — attempts to make the premise of H1 true |
| 5 | [sparse_network_test_progress_v3.md](../legacy/sn_progress/sparse_network_test_progress_v3.md) | v3 design — why "sparsity" is two variables, and the experiment that separates them |

We learned much of §5 to §7 below with difficulty in documents 2 to 4. This document
gives that knowledge as conceptual guidance. The next person then does not find it
again at the same cost.

> **Corrections from v3 (document 5). They replace parts of this file.** §5 speaks of
> "sparsity" as one independent variable. It is two variables: the spikes per *active*
> neuron, and the *silent fraction*. The two variables push in opposite directions, and
> they are confounded at ρ = −0.94 in the no-delay arm of v1. §6a asks you to *close*
> the immune channels. That demand is stronger than the question needs. You can
> *measure and control* the channels instead. v2 used about 8 h to find this. §9 says
> that a synthetic task needs no manipulation. That is true for the *input* only,
> because a network can encode the input timing again as hidden spike counts. Refer to
> document 5, §2 and §3.

---

## 1. The shape of the experiment

```
independent variable  :  hidden-layer sparsity      (set during training)
dependent variable    :  temporal processing        (measured at evaluation)
prediction (H1)       :  sparser ⇒ more temporal ⇒ negative correlation
                         between hidden firing rate and timing-dependence
```

The project already has the measurement device: the **fixed-weight hidden-perturbation
sweep**. You train on clean data. Then you perturb the output of one hidden layer at
evaluation only. Then you look at the decrease in accuracy. Refer to
`knowledge_bank/phase1_investigation.md`. A large decrease shows that the
representation depends on the time of the hidden spikes. A flat curve shows that the
readout is satisfied with a rate code.

Thus the experiment is as follows. Train a family of networks that differ only in the
hidden sparsity. Do the same perturbation sweep on each network. Then correlate the
results.

---

## 2. Why the perturbation must be evaluation-only

This is the most important decision about the protocol. The two protocols answer
different questions:

| Protocol | What it measures | Curve at a hidden site |
|---|---|---|
| **Fixed weight** (train clean → evaluate perturbed) | Does the trained representation *use* the hidden timing? | Informative — flat or steep |
| **Perturbation aware** (train and evaluate at the same perturbation) | Is the task *possible* under the perturbation? | Becomes flat |

With perturbation-aware training, the upstream layer learns to send all the data
through the channel that the perturbation cannot touch. The curve is then flat at all
levels of sparsity. This erases the effect that you look for.

> ⚠️ **Protocol rule:** induce the sparsity **during training**, on clean data. Apply
> the perturbation **only at evaluation**. The two interventions are separate. Do not
> put them in one training run.

---

## 3. How to measure each side

**Sparsity (the independent variable).** Report these values for each trained model, on
the test set:

- **Mean firing rate** — the fraction of active (neuron, time-bin) slots. This is the
  primary x-axis.
- **Spikes per neuron per sample** — the same quantity in an easier form.
- **Spikes per *active* neuron** — the same quantity, but only for the neurons that
  fire.
- **Silent fraction** — the fraction of (sample, neuron) pairs that fire nothing.

The last two values are necessary. A layer at "1.4 spikes per neuron" can be a layer
where each neuron fires 1 to 2 spikes. It can also be a layer where one half of the
neurons are silent and the other half fire 3 to 4 spikes. These are two different
codes, and the primary statistic cannot show the difference. **Always report the
conditional statistic together with the mean.**

**Temporal processing (the dependent variable).** Use the curve of accuracy against
perturbation strength. Decrease the curve to one number that controls for the baseline
accuracy, because sparse networks can start at a lower accuracy:

```
retention      = (acc at max perturbation − chance) / (acc at 0 − chance)
temporal_score = 1 − retention          0 = pure rate code, 1 = fully timing-dependent
```

You must correct for chance **before** you calculate the ratio. If you do not, a
network that has fallen to chance looks like a network with the maximum temporal
score. A more reliable form uses the normalised area below the full curve, and not
only the two end points.

**Always analyse against the *measured* sparsity. Never analyse against the
regularisation setting.** The relation between a penalty control and the sparsity that
you get is nonlinear and seed-dependent. The execution logs show many times that this
relation is different from the intended one.

---

## 4. What a perturbation isolates: the three channels

The spike tensor of a hidden layer can hold class data in three different places:

| Channel | What it is | Does it survive a rate-preserving perturbation? |
|---|---|---|
| **Count** | how many spikes each neuron fires | **yes, exactly** |
| **Identity** | *which* neurons fire | **yes, exactly** |
| **Timing** | *when* the spikes occur | no — the perturbation destroys this channel |

Identity is a special condition of count (the neuron fired, or it did not fire). But
identity has its own name here for two reasons. Stimulus selectivity makes identity,
and identity can be rich even when the counts are almost equal.

Rate-preserving perturbations move the spikes in time. They keep the spike *count* and
the *identity* of each neuron. Examples are per-spike jitter, whole-train shift and
random relocation. This property makes them a clean probe of timing. But this property
also has a result:

> **A perturbation sweep measures how much the readout uses the timing channel *when
> the count channel and the identity channel stay complete*.** If those channels hold
> the answer, the curve stays flat at all levels of sparsity. The flat curve does not
> show that the network does not use timing. It shows that the network never needed
> timing.

---

## 5. The mechanism behind H1 — and the fault in it

The intuitive argument for H1 is as follows:

1. A **dense** layer gives each neuron many spikes. The count of each neuron then has
   resolution. The network can hold the class in a rate code, which a timing
   perturbation cannot touch. The curve is thus flat.
2. A **sparse** layer gives each neuron 0 to 2 spikes. The count then has almost no
   resolution. The only axis that stays is *when* the single spike occurs. This is a
   latency code, and the perturbation destroys it. The curve is thus steep.
3. Therefore sparsity moves the data out of the immune count channel into the weak
   timing channel.

**Step 2 is not correct. This is the central conceptual correction of the full
investigation.** Count resolution is a property of the **population**, and not of one
neuron. One bit for each neuron in a layer of N neurons gives N bits of count data. For
a problem with 20 classes, this is much more than the network needs. A decrease to
"the neuron fires one or two spikes" does *not* remove the ability of the population to
encode the class in the counts.

Two results follow, and we saw both of them:

- **A sparse layer does not close the count channel.** We measured the count channel
  directly. It stayed easy to decode at all levels of sparsity that we reached.
- **Sparsity can *open* the identity channel.** Sparsity can occur because neurons
  become silent for some stimuli and not for other stimuli. The result is stimulus
  selectivity: a labelled-line code. A timing perturbation keeps this code complete,
  and the code becomes *richer* when the sparsity increases. Thus the intervention can
  move the network in the direction opposite to the direction that H1 gives.

Thus H1, as written, mixes "sparse" with "count-uninformative". The two conditions are
different, and the first condition does not cause the second.

---

## 6. What an honest test of H1 needs

There are two conditions. You must **check** both conditions. Do not accept them
without a check.

### 6a. The immune channels must be closed, and you must measure the closure

Sparsity does not close the count channel. Therefore a different mechanism must close
it. You must **measure** the closure before you read any primary correlation:

> **Manipulation check.** Fit a simple linear decoder to predict the label from (i) the
> spike-count vector of each neuron and (ii) the binarised form of that vector. Both
> inputs stay exactly the same under all rate-preserving perturbations. Therefore the
> count decode is a *ceiling* on the accuracy of an immune readout. The count decode
> must fall to a value near chance.

If the count decode does not fall, you cannot interpret the primary result. A flat
curve can mean "the network does not use timing". It can also mean "the network did not
need timing, because the counts were still available". The curve cannot show which of
the two is true. There is a good check for sanity: compare the count decode against the
accuracy of the *network itself*. If the counts decode *better* than the network
scores, the network is clearly not forced to use timing.

Two points about closure, which cost much work:

- **The requirement is almost binary.** A partly narrow count distribution gives very
  little. The decode falls slowly until the counts fall to almost one value. Then the
  decode falls to chance. There is no gradual approach. Therefore "make the penalty a
  little stronger" is not a method.
- **Soft penalties lose the competition.** The count channel can be worth more to a
  network than its own accuracy. The network then uses all the space that a penalty
  leaves: silence, variation inside the band, or a heavy tail. To remove a channel
  reliably, **constrain it in the architecture**. Do not only make it expensive.

### 6b. You must separate timing-specific weakness from general robustness

A network that resists a timing perturbation better can simply resist *all* damage
better. A comparison of timing perturbations against each other cannot find this.

> **Control.** Include a perturbation that destroys the **rate** and not the timing.
> Random spike deletion is an example. If the sparsity trend is also there, the trend
> is general robustness. Remove the control from the timing correlation with a partial
> correlation, or compare the two at equal damage. Do this before you make any claim
> about timing.

This control changed the interpretation of the v1 results by a large amount. Treat the
control as necessary, and not as optional.

---

## 7. Confounds and traps

1. **The escape route through count and identity (§5).** This is the most important
   trap. Check the closure. Do not accept it without a check.
2. **General robustness (§6b).** Always run the control that destroys the rate.
3. **Statistical dilution.** The mean of `spikes per neuron` over the silent neurons
   hides the true code of each neuron. Also report the conditional statistic.
4. **The accuracy confound.** Sparse networks can have a lower clean accuracy. Never
   compare the raw decreases in accuracy. Use the score that is corrected for chance
   and normalised to the baseline. Also show a guard plot of the clean accuracy against
   the sparsity. The plot shows that the trend is not only "sparser networks are worse".
5. **A broken network is not a temporal network.** A network at chance has no
   representation to probe. Remove it from the analysis, and say that you removed it.
6. **Dead neurons.** If you drive the spiking *down*, a neuron can go so far below the
   threshold that its surrogate gradient becomes zero. No penalty can then make it
   active again. To push the activity *up* is the safe direction. With all mechanisms,
   look at the silence from the first epoch. Do not look only at the final value. The
   network can satisfy a constraint early and then release it later, when the task loss
   becomes strong again.
7. **The perturbation magnitude is not equal at different levels of sparsity.** If you
   move a given fraction of many spikes, you disturb a downstream neuron more than if
   you move the same fraction of few spikes. This is a permanent limitation of the
   design. The rate control decreases the problem, but it does not remove it.
8. **A caution about the readout (inherited).** The probe shows only the timing that
   the *readout* uses. A layer can hold temporal structure that the output layer
   ignores. State this limitation. Do not try to correct it here.
9. **The nominal value is not the value that you get.** Analyse the measured
   statistics. Never analyse the control that you set.
10. **Seeds.** Sparsity regularisation interacts strongly with the initialisation. Use
    several seeds, and treat each (setting, seed) as one data point.

---

## 8. What counts as an answer

The milestone is complete when a primary plot exists. The plot shows the
timing-dependence against the measured firing rate. You must **record the manipulation
check and the robustness control with the plot**. Each of these results is legitimate:

| Outcome | Reading |
|---|---|
| Negative correlation, immune channels checked and closed | **H1 supported.** Sparsity moves the code toward timing. |
| Positive correlation, immune channels checked and closed | **H1 refuted on its own terms.** This is the strongest possible negative result. The network had no alternative to timing, but it did not use timing more. |
| No correlation, immune channels checked and closed | The timing-dependence does not change with the sparsity. Sparsity can change the *precision* of a timing code, but not the decision to use one. |
| Correlation of any sign, immune channels **not** closed | You cannot interpret this as a test of H1. It can still be a valid result about the intervention. This is what v1 gave. But you cannot evaluate the claim about the mechanism. |
| Accuracy falls to chance when the immune channels close | H1 is **not testable in this configuration.** The task is not possible with timing only. This is a real result, and you must report it. |

The fourth row and the fifth row are not failures of the experiment. They are results
about the intervention and about the task. Report both of them.

---

## 9. Where to go if this configuration cannot answer the question

There is a difficulty in all of the work. With a natural dataset the count channel is
*useful*. Therefore the network keeps it, and a forced closure can make the task
impossible. A better location for the experiment removes the competition:

**Use a synthetic task in which the inter-spike intervals carry the class data.**
Construct the stimuli so that the spike *counts* hold no class data **by design**. Then
you need no manipulation. The count channel holds nothing, because the data put nothing
in it. The network has no reason to keep the channel. The perturbation sweep then
measures exactly the correct quantity. You also know the temporal signal exactly. This
makes a synthetic task a much stronger location for confirmation than a natural
dataset.

You can vary other axes after the core question has an answer: the perturbation site
(the 2nd hidden layer and the 1st), variants of the dataset, and the presence or the
absence of learnable axonal delays. The delays are a timing mechanism themselves.
Therefore, if you run both conditions, you change a confound into a measurement of how
much of the temporal processing the delays did.
