# Transformer Based Multimodal Trajectory Prediction on the Waymo Open Motion Dataset

A transformer that predicts a road agent's next 8 seconds of motion from 1.1 seconds of history,
its neighbors, the local road graph, and traffic-light states. It produces 32 candidate futures
("modes"), each with a confidence and per-timestep uncertainty, and scores the best 6. Built and trained in Google Colab on the Waymo Open Motion Dataset.

## Approach

- **Data:** 600 shards of the Waymo Open Motion Dataset, split into 480 / 60 / 60 shards, i.e. ~1.05M / 131k / 131k samples.
- **Inputs:** target history plus its 7 nearest neighbors, a 768-point road-graph crop centered 30 m ahead of the agent, and up to 16 traffic lights with their time in state. Everything is in the target's agent frame. Maneuver-grouped anchors, i.e each ground-truth future is labeled with a maneuver group from its geometry (heading change, lateral swing, path length). Anchors are fit by k-means within each group: stationary-reverse / straight / left / right / U-turn.
- **Model (6.4M parameters):** a 6-layer Pre-LN transformer encoder over agent and map tokens, then a 6-layer decoder with 32 mode queries. Each query is conditioned on an anchor trajectory and a learned maneuver-group embedding, and predicts cumulative offsets from its anchor, a per-step sigma and a confidence.
- **Loss:** the ground-truth maneuver group is fixed. Within that group, the slot whose predicted path is closest to the truth gets the Gaussian NLL. The confidence target is zero outside the group and a softmax over in-group path errors inside it, so regression and confidence favour the same slot.
- **Evaluation:** top-6 minADE / minFDE / miss rate on the held-out test split, per agent type, plus Waymo-style metrics (3 / 5 / 8 s, speed-scaled lateral / longitudinal miss boxes), calibration and off-road rate.
- **Training:** AdamW with warmup and cosine decay, bf16, 10-hour budget on one A100.


## Results
| Type | n | minADE (m) | minFDE (m) | Miss rate |
|---|---|---|---|---|
| **All** | 131,465 | **1.27** | **2.94** | **41.8%** |
| Vehicle | 113,960 | 1.36 | 3.14 | 45.1% |
| Pedestrian | 13,847 | 0.62 | 1.34 | 15.4% |
| Cyclist | 3,658 | 1.17 | 2.62 | 37.2% |


## **Outputs**
The outputs below cover different scenarios and both good and bad solutions.
<img src="/output/TEST_turn_like_08.png"/>
<img src="/output/TEST_random_01.png"/>
<img src="/output/TEST_turn_like_02.png"/>
<img src="/output/TEST_turn_like_01.png"/>
<img src="/output/TEST_turn_like_16.png"/>
<img src="/output/TEST_random_15.png"/>
<img src="/output/TEST_random_02.png"/>
<img src="/output/TEST_turn_like_19.png"/>
<img src="/output/TEST_worst_08_minADE_16_4_m.png"/>
<img src="/output/TEST_worst_11_minADE_14_4_m.png"/>

## Iterative Learnings/Notes:
- **Coverage is good.** Across the validation set, all 32 modes get used and are well spread out. The diversity is good.
- **Squiggles** Remaining 26 modes/trajectories and ot smooth, some modes are not getting trained. Initially the squiggles were a lot more with just 10 dataset shards, increasing to 600 helped a lot to smoothen out. Also adding mor road point into to the encder helped reduce the collisions rate.
- **Ranking was the weak spot.** The probabilites were flat in earlier iterations (0.05, 0.04 etc). The confidence head was getting mixed signals. The winner flickered between near identical mdoes. Grouping the maneuvers into differenr intents helped. As well as using a hybrid loss function that using Group + minADE(use endpoint distances before but ADE helped a lot) for Gaussian NLL and just Group for Confidence Cross Entropy. Ranking can still be improved. Maybe move to a different arch using Flow matching to get rid of the ranking and mode.


## Limitations

- Predicts one agent per forward pass, marginal, not joint over the scene.
- Does not predict heading/orientation.
- Trained on a subset of the dataset; the road graph uses point tokens rather than polyline encoders.

## References
1. [MULTIPATH++: EFFICIENT INFORMATION FUSION AND
TRAJECTORY AGGREGATION FOR BEHAVIOR PREDICTION
](https://arxiv.org/pdf/2111.14973)
2. [Wayformer: Motion Forecasting via Simple &
Efficient Attention Networks](https://arxiv.org/pdf/2207.05844)
3. [MTR-VP: Towards End-to-End Trajectory Planning through Context-Driven
Image Encoding and Multiple Trajectory Prediction](https://arxiv.org/pdf/2511.22181)
3. [VectorNet: Encoding HD Maps and Agent Dynamics from
Vectorized Representation](https://arxiv.org/pdf/2005.04259)