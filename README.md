# MRA: Autoregressive Generation-Based Real-Time Production Forecasting with Variable Input Length

## Citation
If you use this code in your research, please cite the following paper: [Xu, Z., & Leung, J. Y. (2024). Dynamic Real-Time Production Forecasting Model for Complex Subsurface Flow Systems with Variable-Length Input Sequences. SPE J.](https://doi.org/10.2118/221482-PA)

## Introduction

Forecasting production time-series for newly drilled wells or those with scant historical production data in a shale gas reservoir is a formidable challenge. This challenge is compounded by the complexities and uncertainties inherent in fractured subsurface systems. Traditional models, which often rely on static features for prediction, fall short as they cannot incorporate the progressively richer insights offered by ongoing production data. To overcome these limitations, we propose the Masked Recurrent Alignment (MRA) methodology, which is based on Autoregressive Generation (AG). MRA utilizes both padding and masking to ensure that all data, regardless of the sequence length, is utilized in the training process, thus addressing the issue of effectively integrating production stream data into the forecasting model.


## Methodology

### Model Structure

MRA employs an encoder-decoder architecture. The encoder encodes historical production information into a vector that initializes the decoder’s hidden state. The decoder then uses this state, combined with current step information, to predict future production data. A key feature of MRA is its ability to iteratively append output to input, facilitated by setting the embedding dimension (sliding window) `m` to `T-1`, and fixing the delay (lag) `d` at 1.

### Segmentation

MRA's design allows for the accommodation of time series of varying lengths within a specified maximum length, enhancing its flexibility. The training set is segmented into distinct time-series segments of lengths ranging from 0 to `T-1`, enlarging the training set to a size of `V×T`, where `V` represents the number of wells. Uniformity in computation is achieved by padding these segments to a uniform length, while masking ensures that padded values are disregarded, thus maintaining data integrity and consistency during training.

The forecasting process is initiated by employing a static feature to predict the first production data point (`x1`). The model then iteratively predicts each subsequent data point (`x2`, `x3`, ..., `xT`) by leveraging the series of previously predicted points (`x1, x2, ..., xn-1`). This methodology generates `T` training pairs for each sequence, cumulatively resulting in `VxT` pairs across the training dataset, where `V` represents the total number of wells involved in the study.

![image](https://github.com/ziming-zx/MRA/assets/55851734/d197187c-2645-4a62-b116-2e8b198f2802)

### Input and Output Shape

The model's input and output structures are designed to accommodate the dynamic nature of the forecasting task:

- **Encoder Input Shape:** The input to the model is formatted as `[VxT, T-1, feature_number]`, where `V` is the number of wells, `T` is the number of time steps in the sequence, and `feature_number` represents the number of features (static, dynamic and autoregressive term) included for each time step.

- **Decoder Input Shape:** The input to the model is formatted as `[VxT, T-1, feature_number]`. Here, `feature_number` represents the number of features (static, dynamic) included for each time step.

- **Output Shape:** The model predicts for `t+1`th timestep value. The output is formatted as `[VxT, 1, 1]`.


## Experiment Setup

The dataset for this study was sourced from public records accessible via PRISM (2023) for a shale gas reservoir in the Central Montney area of British Columbia, Canada, covering 6,154 wells within specific geographical coordinates. Out of these, 2,499 wells with complete data over 36 months were selected for training, and 200 wells were set aside for testing. The model's performance was evaluated using the Root Mean Square Error (RMSE) across these test wells.

![Figure3](https://github.com/ziming-zx/MRA/assets/55851734/1b75969e-686a-40f8-aa17-a294f03047f7)


## Experiment Results

The GRU-MRA model's prediction results are detailed, showcasing the performance across five different training models to illustrate the uncertainty in the training process. For each of the 200 wells in the testing dataset, RMSE was computed, and a distribution of these RMSE values was analyzed for different prediction horizons (`l = 0, 1, 3, and 5 months`). The performance of the models in P10, P50, and P90 cases demonstrates the model's effectiveness in handling variable-length input sequences and underscores the value of incorporating shorter time series segments for improved predictive accuracy, particularly in scenarios with limited training samples.

![Figure5](https://github.com/ziming-zx/MRA/assets/55851734/40cab9b9-8b24-4bde-a89b-1a257cb3b828)


Empirical evidence from this study highlights the superior performance of the proposed models, offering significant advancements in the field of real-time production forecasting.

---------------
Continue reading about the Non-Autoregressive Generation real-time updating method: [MED](https://github.com/ziming-zx/MED)
