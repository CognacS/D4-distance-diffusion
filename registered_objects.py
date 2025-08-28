import src.data.datasets as datasets
import src.data.transforms as transforms
import src.datatypes.features as features
import src.models as models
import src.evaluation as evaluation
import src.callbacks as clb
import src.noise as noise


def main():
    print(datasets.reg_dataresources)
    print(transforms.reg_transforms)
    print(features.reg_features)
    print(models.reg_models)
    print(models.reg_architectures)
    print(evaluation.reg_metrics)
    print(evaluation.reg_assignment)
    print(noise.reg_diffusion)
    print(noise.reg_schedule)
    print(noise.reg_timesampler)
    print(clb.reg_checkpoints)
    print(clb.reg_early_stopping)


if __name__ == '__main__':
    main()