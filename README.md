# Transformer Based Multimodal Trajectory Prediction on the Waymo Open Motion Dataset

An encoder-decoder transformer that predicts a road agent's next 8 seconds of motion from 1.1 seconds of history, its neighbors,
the local road graph and traffic light states. It produces 32 candidate futures ("modes"), each tied to a maneuver
(stationary/reverse, straight, left turn, right turn, Uturn), with a confidence and per timestep uncertainty, and
scores the best 6. Trained on the Waymo Open Motion Dataset.

## Results

Test on 60 shards/131,465 samples. Top-6 of 32 modes. Miss rate is the fixed 2 m endpoint threshold at 8 s.

| Type | n | minADE (m) | minFDE (m) | Miss rate |
|---|---|---|---|---|
| **All** | 131,465 | **1.27** | **2.94** | **41.8%** |
| Vehicle | 113,960 | 1.36 | 3.14 | 45.1% |
| Pedestrian | 13,847 | 0.62 | 1.34 | 15.4% |
| Cyclist | 3,658 | 1.17 | 2.62 | 37.2% |

**Waymo-style metrics** (2 Hz points, speed-scaled lateral/longitudinal miss boxes):

| Horizon | minADE (m) | minFDE (m) | Miss rate |
|---|---|---|---|
| 3 s | 0.35 | 0.63 | 13.2% |
| 5 s | 0.71 | 1.42 | 17.0% |
| 8 s | 1.35 | 3.12 | 24.0% |
| **Average** | **0.80** | **1.72** | **18.1%** |

## Outputs

Each plot shows the top-1 mode, top-6 and all 32 modes. The examples below cover different scenarios, including good and bad predictions.

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

## Approach

- **Data:** 600 shards of the Waymo Open Motion Dataset, split into 480, 60, 60 shards, i.e. ~1.05M, 131k, 131k samples.
- **Inputs:** target history plus its 7 nearest neighbors, a 768 point road graph crop centered 30 m ahead of the agent, and up to 16 traffic lights with their time in state. Everything is in the target's agent frame.
- **Maneuver grouped anchors:** each ground truth future is labeled with a maneuver group from its geometry (heading change, lateral swing, path length). Anchors are fit by k-means within each group giving stationary/reverse, straight, left, right, Uturn.
- **Model(6.4M parameters):** a 6 layer Pre-LN transformer encoder over agent and map tokens, then a 6 layer decoder with 32 mode queries. Each query is conditioned on an anchor trajectory and a learned maneuver group embedding, and predicts cumulative offsets from its anchor, a per-step sigma and a confidence.
- **Loss:** the ground-truth maneuver group is fixed. Within that group, the slot whose predicted path is closest to the truth gets the Gaussian NLL. The confidence target is zero outside the group and a softmax over in group path errors inside it, so regression and confidence favor the same slot.
- **Evaluation:** top-6 minADE/minFDE /miss rate on the held out test split, per agent type, plus Waymo style metrics (3, 5, 8 s, speed-scaled lateral, longitudinal miss boxes), calibration and off road rate.
- **Training:** AdamW with warmup and cosine decay, bf16, 10 hour budget on one A100.

## Repo Structure

The `main.ipynb` is a pipeline designed to run in Google Colab and use google drive for storage.

```
main.ipynb       Run this: data prep -> anchors -> training (optional) -> evaluation -> plots
config.py        All settings (data layout, model, loss, training, evaluation)
data.py          Waymo tf.Example parsing, scene extraction, agent-frame model inputs
prep.py          Download shards, tensorize them to Google Drive, load splits
anchors.py       Maneuver groups and anchor fitting
model.py         TrajectoryTransformer and checkpoint loading
loss.py          Gaussian NLL + confidence loss
metrics.py       minADE / minFDE / miss rate, off road rate, Waymo style metrics
train.py         Training loop with resumable checkpoints
visualize.py     Anchor, scene and prediction plots
checkpoints/     Trained model (trajectory_transformer_best.pt)
output/          Prediction plots on test scenes
Transformer_Based_..._Dataset.ipynb   The original/old single notebook version of this project
```

## Notes

- Coverage is good. Across the validation set, all 32 modes get used and are well spread out, so the predictions are diverse.
- Some of the 26 modes outside the top 6 are not smooth, those modes are not getting trained much. With only 10 dataset shards the squiggles were much worse, and increasing to 600 shards smoothed them out a lot. Adding more road points to the encoder also helped reduce the collision rate.
- Ranking was the weak spot. In earlier iterations the probabilities were flat (0.05, 0.04, ...), the confidence head was getting mixed signals, and the winner flickered between near identical modes. Two changes helped
  - Grouping the maneuvers into different intents.
  - A hybrid loss, group + minADE for the Gaussian NLL, and just the group for the confidence cross entropy. Endpoint distance was used before, ADE helped a lot.
- Ranking can still be improved. One option is a different architecture, such as flow matching, which gets rid of the modes and the ranking step altogether.
- The budgeting was not done properly, the data going in could have been selected with more statistics. Basically amount of map data going in should have been selected carefully.

## Limitations

- Predicts one agent per forward pass, marginal, not joint over the scene.
- Does not predict heading/orientation.
- Trained on a subset of the dataset; the road graph uses point tokens rather than polyline encoders.

## References

1. [MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction](https://arxiv.org/pdf/2111.14973)
2. [Wayformer: Motion Forecasting via Simple & Efficient Attention Networks](https://arxiv.org/pdf/2207.05844)
3. [MTR-VP: Towards End-to-End Trajectory Planning through Context-Driven Image Encoding and Multiple Trajectory Prediction](https://arxiv.org/pdf/2511.22181)
4. [VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation](https://arxiv.org/pdf/2005.04259)
