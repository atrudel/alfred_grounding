# alfred_grounding
Language grounding experiments on the Alfred dataset

## Installation

1. Clone repository
2. Run `install.sh` script
3. Download data
    ```
   cd alfred/data
   sh download_data.sh json_feat
   ```
   
## Training
1. Preprocess the data
   ```bash
   python grounding/data_processing/preprocess_data.py
    ```
2. Launch training
   ```bash
   python grounding/training/train_baseline.py --name experiment_name
   ```

## Evaluation
   ```bash
   python grounding/evaluation/evaluate_objection.py
   ```
