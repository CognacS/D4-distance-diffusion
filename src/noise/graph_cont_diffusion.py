from src.noise import reg_diffusion
from src.noise.schedules import NoiseSchedule

from src.noise.continuous_diffusion import VPGaussianDiffusionProcess
from src.noise.graph_diffusion import GraphDiffusionProcess


@reg_diffusion.register('distance_gaussian')
class DistanceGaussianDiffusionProcess(GraphDiffusionProcess):
    
    def __init__(
            self,
            schedule : NoiseSchedule,
            undirected: bool=True,
            **kwargs
        ):
        """
        Parameters
        ----------
        schedule : DiffusionSchedule
            gives the parameter values for next, sample_t, posterior
        """

        super().__init__(
            edge_dist=VPGaussianDiffusionProcess(schedule),
            undirected=undirected
        )
        
    @property
    def epsilon_parameterization(self) -> bool:
        return self.diffusion_procs_per_data['edge_dist'].epsilon_parameterization