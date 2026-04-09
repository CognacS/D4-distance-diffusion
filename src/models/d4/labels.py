

DENOISE_CE_X = 'denoise_ce_x'
DENOISE_CE_E = 'denoise_ce_e'
DENOISE_CE_C = 'denoise_ce_c'
DENOISE_CE_AUXILIARY_NODE_STATES = 'denoise_ce_auxiliary_node_states'
DENOISE_CE_TOTAL = 'denoise_ce_total'
DENOISE_ACC_X = 'denoise_accuracy_x'
DENOISE_ACC_E = 'denoise_accuracy_e'
DENOISE_ACC_C = 'denoise_accuracy_c'

DENOISE_MSE_DIST = 'denoise_mse_dist'
DENOISE_MAE_DIST = 'denoise_mae_dist'
DENOISE_TOTAL = 'denoise_total'


def denoise_ce_auxiliary_node_state(name: str) -> str:
	return f'denoise_ce_{name}'


def denoise_accuracy_auxiliary_node_state(name: str) -> str:
	return f'denoise_accuracy_{name}'