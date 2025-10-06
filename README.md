# D4: Distance Diffusion for a Truly Equivariant Molecular Design

## Requirements

The code was tested with Python 3.11.7. The requirements can be installed by following these steps:
- Download anaconda/miniconda if needed
- Create a new environment through the given environment files with the following command:
    ```bash
    conda env create -f <env_file>.yml
    ```
    where \<env_file\> is the name of the environment file to use. It is possible to install dependencies for CPU with `environment_cpu.yml` or for GPU with `environment_cuda.yml`.
- Install this package with the following command:
    ```bash
    pip install -e .
    ```
    which will compile required cython and c++ code, if needed.

## Experiments

### Reproducibility
For CUDA>=10.2, to run any experiments in a reproducible way, it is necessary to set the environment variable:
```bash
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
```
or in Windows, for cmd:
```bash
    set CUBLAS_WORKSPACE_CONFIG=:4096:8
```
for PowerShell:
```bash
    $env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
```
and to remove:
```bash
    Remove-Item Env:\CUBLAS_WORKSPACE_CONFIG
```
### Running experiments
Configurations are composed of:
- a `task`, composed of a `dataset` and a `test` suite, defined by an assignment
- a `method`, defined by a `model`, a `trainer`, a `dataloader`, a `datatransform`, and a set of `callbacks`
- a `logger`, e.g., WandB or TensorBoard
- a `platform`, defining the computational resources to use

Presets can be defined in the `config/presets` directory, and experiments can be run by using the following command:
```bash
    python main.py +preset/<group>=<preset> seed=<seed>
```
where `<group>` is the group of experiments, e.g., `final/d4/qm9`, and `<preset>` is the specific preset to use, and `<seed>` is the seed to use for the experiment. The seed is optional, and if not provided, the experiment will be run with a default seed. Experiments are always run with reproducibility.

### Generation
To generate molecules with a trained model, use the following command:
```bash
    python main.py +preset/<group>=<preset> load_ckp=<version> +options=generate
```
This command will load the checkpoint of version `<version>` with the specified configuration `+preset/<group>=<preset>`. More options for customizing the generation, e.g., the number of graphs, can be found in `config/option/generate.yaml`, and can be entered as in [hydra](https://hydra.cc).

### Datasets
Datasets will be downloaded automatically to a new directory ./datasets when running an experiment. Datasets can also be downloaded beforehand using the command:
```bash
    python download_dataset.py +preset/<group>=<preset>
```
which will download the dataset required for configuration `+preset/<group>=<preset>`.

### Checkpoints and logging
Checkpoints are saved in a new directory ./checkpoints, and logging can be done through TensorBoard or WandB, which requires a free account to be used.

### Computing KDEs
To compute Kernel Density Estimates (KDEs) for the generated molecules (in the form of `.pkl` files inside the checkpoints' directories), use the following command:
```bash
    python compute_kdes_checkpoints.py
```
that will generate a new directory `./kdes` with the same folder structure as `./checkpoints`.
Arguments can be entered by customizing `compute_kdes_checkpoints.py`.
To compute KDEs for datasets, use the following command:
```bash
    python compute_kdes_dataset.py +preset/<group>=<preset>
```
which will compute the KDEs for the dataset required for configuration `+preset/<group>=<preset>`.

## License
This code is released under the MIT License. See LICENSE file for details.