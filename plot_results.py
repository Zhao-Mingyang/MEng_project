from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib.pyplot as plt

# Some initial analysis work. Pretty basic here just looking at rewards in evaluation vs. training
# If log interval was increased, could possibly see some more interesting results
# Mostly this file shows how to load results from monitor.csv files



if __name__ == '__main__':
    empty_train_log_dir = "empty_tolerance30/logs"
    empty_eval_log_dir = "empty_tolerance30/logs/eval_log_path"
    obstacles_train_log_dir = "empty_tolerance0/logs"
    obstacles_eval_log_dir = "empty_tolerance0/logs/eval_log_path"
    empty_train_data = load_results(empty_train_log_dir)
    empty_eval_data = load_results(empty_eval_log_dir)
    obstacles_train_data = load_results(obstacles_train_log_dir)
    obstacles_eval_data = load_results(obstacles_eval_log_dir)

    # Plot train vs. eval rewards success rate and error as an example
    plt.figure()
    plt.plot(empty_train_data.t, empty_train_data["r"])
    plt.plot(empty_eval_data.t, empty_eval_data["r"])
    plt.show()

    plt.figure()
    plt.plot(empty_train_data.t, empty_train_data["l"])
    plt.plot(empty_eval_data.t, empty_eval_data["l"])
    plt.show()

    plt.figure()
    plt.plot(empty_train_data.t, empty_train_data["is_success"])
    plt.plot(empty_eval_data.t, empty_eval_data["is_success"])
    plt.show()

    plt.figure()
    plt.plot(empty_train_data.t, empty_train_data["error"])
    plt.plot(empty_eval_data.t, empty_eval_data["error"])
    plt.show()

    plt.figure()
    plt.plot(obstacles_train_data.t, obstacles_train_data["r"])
    plt.plot(obstacles_eval_data.t, obstacles_eval_data["r"])
    plt.show()

    plt.figure()
    plt.plot(obstacles_train_data.t, obstacles_train_data["is_success"])
    plt.plot(obstacles_eval_data.t, obstacles_eval_data["is_success"])
    plt.show()

    plt.figure()
    plt.plot(obstacles_train_data.t, obstacles_train_data["error"])
    plt.plot(obstacles_eval_data.t, obstacles_eval_data["error"])
    plt.show()
