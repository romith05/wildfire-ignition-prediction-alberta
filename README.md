#  Wildfire Ignition Prediction using Deep Learning
## Two-Stage U-Net Pipeline for High-Resolution Geospatial Risk Mapping

###  Overview

Wildfires in Canada are increasing in frequency, intensity, and economic impact. Early rosk prediction of wildfire ignition probability is critical for effective response and mitigation.

This project presents a two-stage deep learning pipeline that predicts wildfire ignition risk at 25m spatial resolution using geospatial and meteorological data.

The system is designed to:
- Maximize ignition detection (recall)
- Minimize false alarms (false positives)
- Produce spatially precise ignition maps

### Key Idea: Two-Stage Architecture
Traditional single-model approaches struggle with:

- High false positives
- Poor spatial precision
- Instability in imbalanced data

This project introduces a cascade pipeline:
#### Stage 1 - Patch-Level Gatekeeper (Model B)
-  Input: 64×64 patch
-  Output: Probability that ignition exists in the patch
-  Purpose: Filter out negative regions

#### Stage 2 — Spatial Refiner (Model A)
-  Input: Only patches flagged by Stage 1
-  Output: Pixel-wise ignition probability map
-  Purpose: Precise localization of ignition

This design separates:
- Detection (Does ignition exist?)
- Localization (Where exactly?)
This architecture is described in detail in the research report.

#### Model Architecture
Both models use U-Net, a convolutional neural network designed for spatial prediction:

-  Encoder–decoder structure
-  Skip connections for spatial detail recovery
-  Pixel-wise prediction using sigmoid output

Training uses:

-  Focal Loss (α≈0.9, γ≈2.5) to handle extreme imbalance
-  Hard negative mining to reduce false positives

#### Data Pipeline
| Category | Features | 
|-------|--------|
| Static | Landcover, DEM, water mask, distance to road, municipalities |
| Weather | Temperature, humidity, wind speed |
| Temporal | sin(month), cos(month) |
| Target | Ignition mask |

Each sample is:
64 × 64 × 17 patch

Dataset includes:
-  Historical wildfire ignition data (Alberta)
-  Balanced ignition and non-ignition samples

### Results
The model is evaluated on fully unseen test data (2021–2023).
| Metric | Value | 
|-------|--------|
| Patch Recall | ~0.945 |
| Pixel Recall | ~0.944 |
| Patch FP Rate| ~0.235 |
| Pixel FP Rate | ~0.00046 |
| Pixel Precision | ~0.20 |

#### Key Insights
-  High recall ensures almost no ignition events are missed
-  Extremely low pixel FP rate ensures clean spatial predictions
-  Moderate patch FP is acceptable due to two-stage filtering
#### Pipeline Flow
![Two stage Pipeline](https://github.com/romith05/wildfire-ignition-prediction-alberta/blob/main/results/plots/wildfire_pipeline.jpg)
#### Sample Outputs
![Alt Text](https://github.com/romith05/wildfire-ignition-prediction-alberta/blob/main/results/plots/sample%201.png)
![Alt Text](https://github.com/romith05/wildfire-ignition-prediction-alberta/blob/main/results/plots/sample%202.png)
![Alt Text](https://github.com/romith05/wildfire-ignition-prediction-alberta/blob/main/results/plots/sample%203.png)
#### Dataset Note
Full dataset is not included due to size.
#### Future Work
- Temporal modeling (multi-day weather sequences)
- Real-time API integration (Meteostat / ECCC)
- Larger spatial context (beyond patches)
- Satellite-based anomaly detection integration
- Cloud deployment (AWS + Docker + Streamlit dashboard)
#### Author
**Romith Bondada**
MSc Data Science — University of Calgary
