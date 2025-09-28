from src.kdes import scan_all_checkpoints_and_compute_kdes

def main():
    kde_kwargs = {}
    scan_all_checkpoints_and_compute_kdes(kde_kwargs=kde_kwargs)

if __name__ == '__main__':
    main()
