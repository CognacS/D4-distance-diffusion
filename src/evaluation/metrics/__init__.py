from src.evaluation.metrics.core import Metric

############################  COMPUTATIONAL METRICS  ###########################

# Training time performance metrics
KEY_COMPUTATIONAL_TRAIN_TIME = 'train_time'
KEY_COMPUTATIONAL_TRAIN_MEMORY = 'train_memory'


#####################  PREDICTION OR MODEL RELATED METRICS  ####################

KEY_NLL = 'nll'


##############################  SAMPLING METRICS  ##############################
# the following metrics are computed by generating new samples
# with the model, which can be time consuming

# VUN metrics (Validity-Uniqueness-Novelty) on molecules
KEY_MOLECULAR_VALIDITY = 'molecular_validity'
KEY_MOLECULAR_UNIQUENESS = 'molecular_uniqueness'
KEY_MOLECULAR_NOVELTY = 'molecular_novelty'
KEY_BOND_DISTANCE = 'bond_distance'
KEY_EDGE_TYPES_DISTRIBUTION = 'edge_types_distribution'

# number of connected components (on molecules, needs validity metric)
KEY_MOLECULAR_CONN_COMP = 'molecular_conn_comp'

# distributional metrics on molecules
KEY_FCD = 'fcd'
KEY_NSPDK = 'nspdk' # not really molecular, but still used


# VUN metrics (Validity-Uniqueness-Novelty) on graphs
KEY_GRAPH_UNIQUE = 'graph_uniqueness'
KEY_GRAPH_UNIQUE_NOVEL = 'graph_uniqueness_novelty'
KEY_GRAPH_VUN = 'graph_vun'

# distributional metrics on graphs
KEY_GRAPH_DEGREE = 'graph_degree'
KEY_GRAPH_SPECTRE = 'graph_spectre'
KEY_GRAPH_CLUSTERING = 'graph_clustering'
KEY_GRAPH_ORBIT = 'graph_orbit'
KEY_GRAPH_GIN = 'graph_gin'

KEY_GRAPH_NODES = 'graph_nodes'
KEY_GRAPH_ECCENTRICITY = 'graph_eccentricity'

# number of connected components (on networkx graphs)
KEY_GRAPH_CONN_COMP = 'graph_conn_comp'

# distributional metrics on graphs as computed by CDGS
KEY_GRAPH_CDGS = 'graph_cdgs'

# computatinal time and memory
KEY_SAMPLING_TIME = 'sampling_time'
KEY_SAMPLING_MEMORY = 'sampling_memory'


############################  SUMMARIZING METRICS  #############################

KEY_WEIGHTED_SUM_METRIC = 'weighted_sum_metric'