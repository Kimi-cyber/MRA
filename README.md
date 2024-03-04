# MRA: Autoregressive Generation-Based Real-Time Production Forecasting with Variable Input Length

Masked Recurrent Alignment (MRA) is a novel approach designed for continuous prediction using limited production history. It leverages autoregressive generation (AG) for sequence-to-sequence (seq2seq) generation, while its masking capability enables the handling of sequences with variable input lengths.

## Introduction

Forecasting production time-series for newly drilled wells or those with scant flow and pressure historical data is a formidable challenge. This challenge is compounded by the complexities and uncertainties inherent in fractured subsurface systems. Traditional models, which often rely on static features for prediction, fall short as they cannot incorporate the progressively richer insights offered by ongoing production data. To overcome these limitations, we propose the Masked Recurrent Alignment (MRA) methodology, which is based on autoregressive generation (AG). MRA utilizes both padding and masking to ensure that all data, regardless of the sequence length, is utilized in the training process, thus addressing the issue of effectively integrating production stream data into the forecasting model.

## Methodology

### Model Structure

MRA is modeled on the structure of the Transformer and employs an encoder-decoder architecture. The encoder encodes historical production information into a vector that initializes the decoder’s hidden state. The decoder then uses this state, combined with current step information, to predict future production data. A key feature of MRA is its ability to iteratively append output to input, facilitated by setting the embedding dimension (sliding window) `m` to `T-1`, and fixing the delay (lag) `d` at 1.

### Segmentation

MRA's design allows for the accommodation of time series of varying lengths within a specified maximum length, enhancing its flexibility. The training set is segmented into distinct time-series segments of lengths ranging from 0 to `T-1`, enlarging the training set to a size of `V×T`, where `V` represents the number of wells. Uniformity in computation is achieved by padding these segments to a uniform length, while masking ensures that padded values are disregarded, thus maintaining data integrity and consistency during training.

### Input/Output Shape

The model starts by using a static feature to predict the first production data point (`x1`). Subsequent points (`x2`, `x3`, ..., `xT`) are predicted by iteratively combining the previously predicted points (`x1, x2, ..., xn-1`). This generates `T` training pairs for a sequence, resulting in `VxT` pairs for the training dataset, where `V` is the number of wells.

## Experiment Setup

The dataset for this study was sourced from public records accessible via PRISM (2023) for a shale gas reservoir in the Central Montney area of British Columbia, Canada, covering 6,154 wells within specific geographical coordinates. Out of these, 2,499 wells with complete data over 36 months were selected for training, and 200 wells were set aside for testing. The model's performance was evaluated using the Root Mean Square Error (RMSE) across these test wells.

## Experiment Results

The GRU-MRA model's prediction results are detailed, showcasing the performance across five different training models to illustrate the uncertainty in the training process. For each of the 200 wells in the testing dataset, RMSE was computed, and a distribution of these RMSE values was analyzed for different prediction horizons (`l = 0, 1, 3, and 5 months`). The performance of the models in P10, P50, and P90 cases demonstrates the model's effectiveness in handling variable-length input sequences and underscores the value of incorporating shorter time series segments for improved predictive accuracy, particularly in scenarios with limited training samples.

Empirical evidence from this study highlights the superior performance of the proposed models, offering significant advancements in the field of real-time production forecasting.
