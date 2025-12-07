# Disclaimer

This fork represents a slighlty modified version of the original code, specifically created for the tutorial "Graph Machine Learning with Missing Multimodal Information", held at the 4th Learning on Graphs Conference (LoG 2025). Website: https://log-centralesupelec.github.io/missing-multimod-gml-log2025/.

Author: [Daniele Malitesta](https://danielemalitesta.github.io)

# Multimodal Patient Representation Learning with Missing Modalities and Labels (ICLR 2024)

This repository contains code for the ICLR'24 paper: [Multimodal Patient Representation Learning with Missing Modalities and Labels](https://openreview.net/pdf?id=Je5SHCKpPa).

## Dependencies

```
python==3.8.18
torch==2.0.1
```

## Repository Structure

- `src/`: Source code for MedLink
    - `preprocess/`: Scripts for data preprocessing
    - `dataset/`: Data, Dataset, Tokenizer, Vocabulary, and collate_fn
    - `core/`: Core implementation for the MUSE method
    - `metrics.py`: Metrics for model evaluation
    - `helper.py`: Helper class for model training, evaluation, and inference
    - `utils.py`: Utility functions

## How to Reproduce

Follow these steps to reproduce the results:

1. Edit the path in `src/utils.py` to your local path.
2. Obtain the eICU and MIMIC-IV datasets and place it under `{raw_data_path}`.
3. Run the following notebooks under `src/preprocess` in the specified order to prepare the data:
   1. eICU:
      1. Run `parse_eicu_remote.ipynb`
      2. Run `preprocess_eicu.py`
      3. Run `build_vocab_eicu.py`
      4. Run `data_split_eicu.py`
   2. MIMIC-IV:
      1. Run `parse_mimic4_remote.ipynb`
      2. Run `preprocess_mimic4.py`
      3. Run `build_vocab_mimic4.py`
      4. Run `data_split_mimic4.py`
   3. Run `get_code_embeddings.py`
4. Execute `run.py` under `src/core` to train the model:
   ```
   python run.py \
   --dataset [mimic4/eicu] \
   --task [mortality/readmission] \
   --official_run
   ```

## Citation

```
@inproceedings{
wu2024multimodal,
title={Multimodal Patient Representation Learning with Missing Modalities and Labels},
author={Zhenbang Wu and Anant Dadu and Nicholas Tustison and Brian Avants and Mike Nalls and Jimeng Sun and Faraz Faghri},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=Je5SHCKpPa}
}
```
