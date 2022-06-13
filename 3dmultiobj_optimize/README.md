#  Multi-Object 3D Scene Representations Network

![teaser](/3dmultiobj_optimize/imgs/teaser.png)


## Pre-Training of DeepSDF
1) Training

```bash
python train_deepsdf.py  --message <ExperimentName> --model <Model> --config <ConfigFile> --log_dir <LogDir> --dataset <DatasetName> --data_dir <DataDir>  

```

2) Evaluation

```bash
python eval_deepsdf.py --message <ExperimentName> --model <Model> --config <ConfigFile> --log_dir <LogDir> --dataset <DatasetName> --data_dir <DataDir> --split <'val'/'train'>

```

## Training of MultiObj3DNet

The new loss functions are implemented in models/mosnet.py, their weights in training can be set in cnfg.py

1) Training

```bash
python train.py --message <ExperimentName> --model <Model> --config <ConfigFile> --log_dir <LogDir> --dataset <DatasetName> --data_dir <DataDir>

```

2) Evaluation

```bash
python eval.py --message <ExperimentName> --model <Model> --config <ConfigFile> --log_dir <LogDir> --dataset <DatasetName> --data_dir <DataDir> --split <'val'/'train'>

```

## Render-and-Compare Optimization

```bash
python render-and-compare.py --dataset <DatasetName> --data_dir <DataDir> --model <Model> --config <ConfigFile> --message <ExperimentName> --split <Split> --start_scene <StartScene> --stop_scene <StopScene>

```
Here start_scene and stop_scene define a range of scenes to be optimized in the dataset.
The detailed configuration of the optimization happens inside CONFIG['r-and-c'] in render-and-compare.py.

----------------------------------------------------------------------------------------------

## Preprocessing of Data (OLD)

1) use modiefied clevr code to generate data
2) brightening of images (+ can be used for rescaling), convert blender output mesh into triangular mesh (naiv approach)
```bash
python ./preprocess/preprocess-clevr_prepare.py --data_dir <DATA_DIR>
```
```bash
python ./preprocess/preprocess-shapenet_prepare.py --data_dir <DATA_DIR>
```
(As ShapeNet data can now also be rendered using the clevr code, pre-processing with the first script should be possible here as well.)
3) 
```bash
python ./preprocess/prepare_meshes.py --input  <DATA_DIR>/objects/triangular
```
4) Generate Training Data
- SDF Samples
```bash
python ./preprocess/sample_meshes.py --input  <DATA_DIR>/objects/scaled
```
Slices (Visualization)
```bash
python ./preprocess/slice_meshes.py --input  <DATA_DIR>/objects/scaled
```
