import sys

from fingerprint import Fingerprint
from algo import replay_scenario, analyse_scenario_result, ml_based
from algo import simple_eckersley, rule_based, split_data, train_ml, optimize_lambda
from utils import get_consistent_ids, get_fingerprints_experiments
from algo import benchmark_parallel_f_ml, benchmark_parallel_f_rules, parallel_pipe_task_ml_f, parallel_pipe_task_rules_f
import MySQLdb as mdb
from functools import partial

# Modes to run main.py
CONSISTENT_IDS = "getids"
REPLAY_ECKERSLEY = "replayeck"
AUTOMATE_REPLAYS = "auto"
RULE_BASED = "rules"
ML_BASED = "ml"
AUTOMATE_ML = "automl"
AUTO_DEEP_EMBEDDING = "deepembedding"
OPTIMIZE_LAMBDA = "lambda"
BENCHMARK_ML = "automlbench"
BENCHMARK_RULES = "autorulesbench"

VISIT_FREQUENCIES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20]


def fetch_consistent_user_ids(cur):
    print("Fetching consistent user ids.")
    user_id_consistent = get_consistent_ids(cur)
    with open("./data/consistent_extension_ids.csv", "w") as f:
        f.write("user_id\n")
        for user_id in user_id_consistent:
            f.write(user_id + "\n")


def automate_replays(cur, exp_name, algo_matching_name, nb_min_fingerprints):
    exp_name += "-%s-%d" % (algo_matching_name, nb_min_fingerprints)

    attributes = Fingerprint.INFO_ATTRIBUTES + Fingerprint.HTTP_ATTRIBUTES + \
                 Fingerprint.JAVASCRIPT_ATTRIBUTES + Fingerprint.FLASH_ATTRIBUTES

    print("Begin automation of scenarios")
    print("Start fetching fingerprints...")
    fingerprint_dataset = get_fingerprints_experiments(cur, nb_min_fingerprints, attributes)
    train_data, test_data = split_data(1, fingerprint_dataset)
    print("Fetched %d fingerprints." % len(fingerprint_dataset))
    print("Length of test set: {:d}".format(len(test_data)))

    if algo_matching_name == "hybrid_algo":
        model = train_ml(fingerprint_dataset, train_data, load=False)

    # select the right algorithm
    algo_name_to_function = {
        "eckersley": simple_eckersley,
        "rulebased": rule_based,
        "hybrid_algo": partial(ml_based, model=model, lambda_threshold=0.1)
    }
    algo_matching = algo_name_to_function[algo_matching_name]

    # we iterate on different values of visit_frequency
    for visit_frequency in VISIT_FREQUENCIES:
        result_scenario = replay_scenario(test_data, visit_frequency,
                                          algo_matching,
                                          filename="./results/" + exp_name + "_" + str(
                                              visit_frequency) + "scenario_replay_result.csv")
        analyse_scenario_result(result_scenario, test_data,
                                fileres1="./results/" + exp_name + "_" + str(visit_frequency) + "-res1.csv",
                                fileres2="./results/" + exp_name + "_" + str(visit_frequency) + "-res2.csv",
                                )


def automate_ml_embedding(cur, exp_name, nb_min_fingerprints):
    print("Start automating ml based scenario")



def optimize_lambda_main_call(cur):
    attributes = Fingerprint.INFO_ATTRIBUTES + Fingerprint.HTTP_ATTRIBUTES + \
                 Fingerprint.JAVASCRIPT_ATTRIBUTES + Fingerprint.FLASH_ATTRIBUTES

    nb_min_fingerprints = 6
    print("Start fetching fingerprints...")
    fingerprint_dataset = get_fingerprints_experiments(cur, nb_min_fingerprints, attributes)
    print("Fetched %d fingerprints." % len(fingerprint_dataset))
    train_data, test_data = split_data(0.4, fingerprint_dataset)
    optimize_lambda(fingerprint_dataset, train_data, test_data)


def benchmark_ml(cur, prefix_files, nb_cores):
    nb_processes = [1, 2, 4, 8, 16, 24, 32]
    nb_fingerprints = [500000, 1000000, 2000000]
    #  nb_fingerprints = [500000, 1000000, 2000000]
    fn = parallel_pipe_task_ml_f
    with open("./benchres/%s.csv" % prefix_files, "w")as f:
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n" %
                ("nb_fingerprints",
                 "nb_cores",
                 "nb_processes",
                 "avg",
                 "max",
                 "min",
                 "median",
                 "q1",
                 "q3")
                )
        for nb_fingerprint in nb_fingerprints:
            for nb_process in nb_processes:
                mean, min, max, p25, p50, p75 = benchmark_parallel_f_ml(fn, cur, nb_fingerprint, nb_process)
                f.write("%d,%d,%d,%f,%f,%f,%f,%f,%f\n" % (
                    nb_fingerprint,
                    nb_cores,
                    nb_process,
                    mean,
                    max,
                    min,
                    p50,
                    p25,
                    p75
                ))


def benchmark_rules(cur, prefix_files, nb_cores, nb_processes=[1, 2, 4, 8, 16, 24, 32],
                    nb_fingerprints=[500000, 1000000, 2000000]):
    fn = parallel_pipe_task_rules_f
    with open("./benchres/%s.csv" % prefix_files, "w")as f:
        f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s\n" %
                ("nb_fingerprints",
                 "nb_cores",
                 "nb_processes",
                 "avg",
                 "max",
                 "min",
                 "median",
                 "q1",
                 "q3")
                )
        for nb_fingerprint in nb_fingerprints:
            for nb_process in nb_processes:
                mean, min, max, p25, p50, p75 = benchmark_parallel_f_rules(fn, cur, nb_fingerprint, nb_process)
                f.write("%d,%d,%d,%f,%f,%f,%f,%f,%f\n" % (
                    nb_fingerprint,
                    nb_cores,
                    nb_process,
                    mean,
                    max,
                    min,
                    p50,
                    p25,
                    p75
                ))
def main(argv):
    con = mdb.connect(host="127.0.0.1", port=3306, user="stalker", passwd="baddy", db="canvas_fp_project")
    cur = con.cursor(mdb.cursors.DictCursor)
    if argv[0] == CONSISTENT_IDS:
        fetch_consistent_user_ids(cur)
    elif argv[0] == AUTOMATE_REPLAYS:
        automate_replays(cur, argv[1], argv[2], int(argv[3]))
    elif argv[0] == AUTOMATE_ML:
        automate_replays(cur, argv[1], "hybrid_algo", int(argv[2]))
    elif argv[0] == OPTIMIZE_LAMBDA:
        optimize_lambda_main_call(cur)
    elif argv[0] == BENCHMARK_ML:
        benchmark_ml(cur, argv[1], int(argv[2]))
    elif argv[0] == BENCHMARK_RULES:
        benchmark_rules(cur, argv[1], int(argv[2]))
    elif argv[0] == AUTO_DEEP_EMBEDDING:
        automate_ml_embedding(cur, argv[1], int(argv[2]))

if __name__ == "__main__":
    main(sys.argv[1:])
