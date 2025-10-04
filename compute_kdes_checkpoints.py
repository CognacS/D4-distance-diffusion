from src.kdes import scan_all_checkpoints_and_compute_kdes, aggregate_statistics_of_configs

def main():
    kde_kwargs = {}
    print('Scanning all checkpoints and computing KDEs...')
    scan_all_checkpoints_and_compute_kdes(kde_kwargs=kde_kwargs)
    print('Aggregating statistics of atom and bond types across all configurations...')
    aggregate_statistics_of_configs(round_digits=4, latex_format=True)

if __name__ == '__main__':
    main()
