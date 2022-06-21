## BEV Benchmark Evaluation

The evaluation code of BEV is integrated into `simple-romp`, `evaluation` folder.

To run the evaluation, please first install simple-romp following [the guidance](../simple_romp/README.md).  
To prepare the evaluation datasets, please refer to [dataset.md](../docs/dataset.md).  
In each evaluation code, please set the dataset path and `output_save_dir` for saving the predictions.

### Relative Human

Please properly set the path in `simple_romp/evaluation/eval_Relative_Human.py` and then run  
```
cd simple_romp/evaluation
python eval_Relative_Human.py
```
You can also download the [predictions](https://github.com/Arthur151/Relative_Human/releases/download/Predictions/all_results.zip).

### AGORA

Please properly set the path in `simple_romp/evaluation/eval_AGORA.py` and then run  
```
cd simple_romp/evaluation
python eval_AGORA.py
```

### CMU Panoptic

Please properly set the path in `simple_romp/evaluation/eval_cmu_panoptic.py` and then run  
```
cd simple_romp/evaluation
python eval_cmu_panoptic.py
```
You can also download the [predictions](https://github.com/Arthur151/ROMP/releases/download/V2.1/cmu_panoptic_predictions.npz) and then set the results path in `evaluation_results` function to get the matrix numbers. 
