# Transformer Based Multimodal Trajectory Prediction on the Waymo Open Motion Dataset

A transformer that predicts a road agent's next 8 seconds of motion from 1.1 seconds of history,
its neighbors, the local road graph, and traffic-light states. It produces 32 candidate futures
("modes"), each with a confidence and per-timestep uncertainty, and scores the best 6. Built and trained in Google Colab on the Waymo Open Motion Dataset.

## Approach

- **Data:** 150 shards of the Waymo Open Motion Dataset, split by shard into train / val / test
  (120 / 15 / 15), ~108k / 13.6k / 13.8k samples.
- **Inputs:** target history plus 8 nearest neighbors, a 768-point road-graph crop centered 30 m
  ahead of the agent, and up to 16 traffic lights. Everything is in the target's agent frame.
- **Model (~6.4M params):** a 6-layer Pre-LN transformer encoder over agent and map tokens, then a
  6-layer decoder with 32 mode queries. Each query is conditioned on a k-means anchor trajectory and
  predicts a residual offset from that anchor (MultiPath style) plus a confidence.
- **Loss:** Gaussian NLL on the single closest-by-path mode (winner-take-all approach) plus cross-entropy on
  the confidences. The predicted standard deviation is floored at ~0.2 m.
- **Selection:** the 6 scored modes are chosen by endpoint non-maximum suppression (MTR style):
  walk modes in descending confidence and skip any whose 8 s endpoint is within 2.5 m of one already
  kept, so near-duplicates don't fill all six slots.
- **Evaluation:** minADE / minFDE / miss rate on the held-out test split, broken down by agent type.
- **Training Env** 6 hrs on Google Colab with A100 GPU

## Configuration

| | |
|---|---|
| Modes / scored | 32 / top-6 (endpoint NMS, 2.5 m) |
| Model dim / heads / layers | 192 / 4 / 6 encoder, 6 decoder |
| Feed-forward dim / dropout | 768 / 0.1 |
| Assignment | trajectory (closest-ADE), winner-take-all |
| Uncertainty floor / ceiling | log-std ∈ [−1.6, 2.0] (~0.2 m–7.4 m) |
| NLL warmup / label smoothing | 10 epochs / 0.1 |
| Batch / peak LR / schedule | 32 / 3e-4 / cosine to 1e-6, 5-epoch warmup |
| Weight decay / grad clip | 1e-4 / 1.0 |
| Horizon | 1.1 s history → 8 s future (80 steps) |

## Results
| Type | n | minADE (m) | minFDE (m) | Miss rate |
|---|---|---|---|---|
| **All** | 13,832 | **1.60** | **3.86** | **47.4%** |
| Vehicle | 13,409 | 1.63 | 3.93 | 48.1% |
| Pedestrian | 356 | 0.70 | 1.60 | 20.8% |
| Cyclist | 67 | 1.36 | 3.06 | 41.8% |


**Outputs**
<img src="/output/TEST_random_01.png"/>
<img src="/output/TEST_random_02.png"/>
<img src="/output/TEST_turn_like_18.png"/>
<img src="/output/TEST_random_12.png"/>
<img src="/output/TEST_turn_like_13.png"/>
<img src="/output/TEST_random_00.png"/>
<img src="/output/TEST_random_09.png"/>
<img src="/output/TEST_random_11.png"/>
<img src="/output/TEST_random_16.png"/>

## Key finding

- **Coverage is good.** Across the validation set, all 32 modes get used and are well spread out. The diversity is good.
- **Ranking is the weak spot.** If the model could always keep its own single
  best-fitting mode, minADE would fall from 1.60 m to about 0.85, so roughly 0.78 m is lost
  purely to poor ranking, not to bad predictions. The best-fitting mode is bumped out of the scored
  top-6 about half the time.
- **The head learned the prior, not the posterior.** Confidence correlates almost perfectly with how
  often a mode wins overall, but per scene the top-confidence mode is the best one only
  ~1 in 5 times. In other words the head ranks by which maneuvers are common in general (going
  straight) rather than which fits this scene, so on a turning agent the correct turn mode exists
  but ranks below the popular straight modes and gets cut.
- Re-tuning the selection rule (NMS distance, plain top-k) only
  slides along a minADE-vs-miss-rate trade-off; it never approaches the 0.85 m ceiling. The limit is
  in the confidence scores themselves.
- The setup here (winner-take-all training, top-6 with
  de-duplication, a confidence head) resulting in  poorly-calibrated confidence.
- Visually the top modes render smooth while some modes are squiggly. That's a direct
fingerprint of winner-take-all training- only the winning mode is trained on each example, so
rarely-winning modes are undertrained.

## What worked, and what didn't

- **Soft assignment (early).** Splitting the training signal across all modes collapsed the
  confidence head, every mode looked equally likely. Abandoned.
- **Endpoint assignment, 6 modes, no Traffic lights.** Train only the mode whose endpoint is closest to
  truth. Test: 1.67 / 3.93 / 68.4%. Good closeness, far too many misses.
- **Anchor assignment.** Forcing each mode to its fixed k-means shape regressed badly
  (~2.71 val) and overfit; the confidence head saturated. A clean negative result.
- **Trajectory assignment, 32 modes, traffic lights, top-6 NMS (run 3, 100 shards).** Train the mode
  whose whole path is closest. Test: 1.85 / 4.20 / 54.0%. Miss rate dropped sharply (68% → 54%).
- **uncertainty floor, the ~best model.** Validation loss had been drifting up
  while minADE improved: the model was shrinking its stated uncertainty to game the loss without
  predicting better. Flooring the standard deviation at ~0.2 m removed the drift and produced the
  1.60 / 3.86 / 47.4% model above. Because only that one setting changed on the same 150-shard data,
  it's a clean before/after that isolates the fix.

## Limitations

- Predicts one agent per forward pass, marginal, not joint over the scene.
- Does not predict heading/orientation.
- Confidence ranking is weak.
- Trained on a subset of the dataset; the road graph uses point tokens rather than polyline encoders.