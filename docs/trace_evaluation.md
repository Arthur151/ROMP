## TRACE Benchmark Evaluation

The evaluation code of TRACE is integrated into `simple-romp`, `trace/eval.py`, `trace/evaluation` folder.

To run the evaluation, please first install simple-romp following [the guidance](../simple_romp/README.md).  
To prepare the evaluation datasets, please refer to [trace_dataset.md](../docs/trace_dataset.md).  

Please set the path to datasets in **dataset_dir of `simple-romp/trace/eval.py`**, and then run:

### DynaCam
For details of DynaCam dataset, please refer to [[DynaCam Dataset]](https://github.com/Arthur151/DynaCam), which directly provides [predictions](https://github.com/Arthur151/DynaCam/releases/tag/predictions), and code for evaluation and visualization. 

You may also evaluate on DynaCam via running
```
cd simple_romp/trace
python -m eval --eval_dataset=DynaCam
```

### MuPoTS
```
cd simple_romp/trace
python -m eval --eval_dataset=mupots
```

### Dyna3DPW
```
cd simple_romp/trace
python -m eval --eval_dataset=Dyna3DPW
```

### 3DPW
```
cd simple_romp/trace
python -m eval --eval_dataset=3DPW
```