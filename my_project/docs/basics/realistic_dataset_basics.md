# SHD and SSC Dataset Modifications

## Why Modify?

SHD/SSC contain strong **rate-based cues** — spike count vectors differ significantly between classes, so even a simple MLP trained only on spike counts achieves ~50% on SHD. This makes it impossible to tell whether an SNN is genuinely using spike timing or just exploiting firing rates.

---

## Stage 1: Whole → Part (Neuron & Sample Filtering)

**Problem:** Some neurons fire zero spikes in certain samples, making spike count normalization impossible.

**What's done:**
1. For each neuron *i*, compute its minimum spike count across all samples: $c_i^{\min} = \min_m \sum_t x_{m,i,t}$
2. If $c_i^{\min} < \theta$ (threshold $\theta = 2$), the neuron is a candidate for removal
3. **But first**, check if the low count is caused by only a few problematic samples: define $\mathcal{M}_i$ = samples where neuron *i* fires fewer than $\theta$ times
4. If $|\mathcal{M}_i|/M < \epsilon$ ($\epsilon = 0.01$), **remove those samples instead of the neuron** — this retains more neurons
5. Finally, downsample each class to match the smallest class size for fair comparison

**Result:** A filtered dataset where every retained neuron fires at least $\theta$ times in every sample.

---

## Stage 2: Part → Norm (Min-Count Spike Normalization)

**What's done:**
- For each neuron *i*, randomly subsample each sample's spike train to retain exactly $c_i'^{\min}$ spikes (the minimum count across filtered samples)
- Every neuron now fires the **same fixed number of spikes** across all samples

**Result:** All rate information is eliminated — only **spike timing** carries class information.

**Trade-off:** The random subsampling may distort ISI structure or cross-neuron synchrony depending on which spikes survive, making Norm a strict/conservative test of timing-based learning.

---

## Summary Table

| Variant | Neurons | Spike counts | Rate info |
|---------|---------|--------------|-----------|
| Whole | All 700 | Variable | Yes |
| Part | Filtered (low-activity removed) | Variable | Yes |
| Norm | Same as Part | Fixed per neuron | **None** |
