def get_save_dir_name(args):
    # Define the default values as per argparse
    defaults = {
        'loss': 'mse',
        'batch_size': 128,
        'lr': 1e-4,
        'lambda_r': 1e-4,
        'lambda_b': 5,
        'num_linear_epochs': 3,
        'num_top_epochs': 20,
        'num_full_epochs': 0,
        'noisy_frac': 0,
        'seed': 42
    }

    # Start with the loss name
    parts = [args.loss]

    # Iterate over args and compare with defaults
    for key, default_value in defaults.items():
        value = getattr(args, key)
        if value != default_value and key != 'loss':  # Skip 'loss' since it's already added
            parts.append(f"{key}={value}")

    return '_'.join(parts)

def score_function(y, kde, delta=1e-5):
        # Derivative of log density
        log_p_plus = kde.logpdf(y + delta)[0]
        log_p_minus = kde.logpdf(y - delta)[0]
        d_logp = (log_p_plus - log_p_minus) / (2 * delta)
        return d_logp