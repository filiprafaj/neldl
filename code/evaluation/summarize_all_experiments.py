import argparse
import os
from pathlib import Path


def file_is_used(filepath):
    from subprocess import DEVNULL, PIPE, STDOUT, Popen, check_output
    try:
        lsout = Popen(['lsof', filepath], stdout=PIPE, shell=False, stderr=DEVNULL)
        check_output(["grep", filepath], stdin=lsout.stdout, shell=False)
        return True
    except:
        # check_output will throw an exception here
        # if it won't find any process using that file
        return False


def process_experiment(args, goldspans_micro, allspans_micro, goldspans_macro, allspans_macro, log_file):

    with open(log_file, "r") as fin:
        print("Processing file: ", log_file)
        best = dict()
        best["goldspans_dev_f1_micro"] = 0
        best["allspans_dev_f1_micro"] = 0
        best["goldspans_test_f1_micro"] = 0
        best["allspans_test_f1_micro"] = 0
        best["goldspans_dev_f1_macro"] = 0
        best["allspans_dev_f1_macro"] = 0
        best["goldspans_test_f1_macro"] = 0
        best["allspans_test_f1_macro"] = 0

        mode = ""
        for line in fin:
            line = line.rstrip()
            if line.startswith("EVALUATION"):
                eval_cnt = line[line.rfind(' ')+1:]
            elif line.startswith("Mode: goldspans"):
                mode = "goldspans"
            elif line.startswith("Mode: allspans"):
                mode = "allspans"
            elif line.startswith(args.dev_set): # "aida_dev"
                try:
                    micro_line = next(fin)
                    macro_line = next(fin)

                  # FIND BEST (macro or micro)
                    current_f1_1 = (float(micro_line.split()[-1]) if args.max_micro_or_macro=='micro'
                                  else float(macro_line.split()[-1]))
                    current_f1_2 = (float(micro_line.split()[-1]) if args.max_micro_or_macro=='macro'
                                  else float(macro_line.split()[-1]))
                    best_f1_1 = (best[mode+"_dev_f1_micro"] if args.max_micro_or_macro=='micro'
                               else best[mode+"_dev_f1_macro"])
                    best_f1_2 = (best[mode+"_dev_f1_micro"] if args.max_micro_or_macro=='macro'
                               else best[mode+"_dev_f1_macro"])

                    if current_f1_1 >= best_f1_1:
                        if ((current_f1_1 > best_f1_1) or (current_f1_2 >= best_f1_2)):
                            best[mode+"_eval_cnt"] = eval_cnt
                            # micro
                            best[mode+"_dev_f1_micro"] = float(micro_line.split()[-1])
                            best[mode+"_dev_pr_micro"] = float(micro_line.split()[2])
                            best[mode+"_dev_re_micro"] = float(micro_line.split()[4])
                            # macro
                            best[mode+"_dev_f1_macro"] = float(macro_line.split()[-1])
                            best[mode+"_dev_pr_macro"] = float(macro_line.split()[2])
                            best[mode+"_dev_re_macro"] = float(macro_line.split()[4])

                except StopIteration:
                    break

            elif line.startswith(args.test_set): # "aida_test"
                try:
                    micro_line = next(fin)
                    macro_line = next(fin)

                    if best[mode+"_eval_cnt"] == eval_cnt:
                      # micro
                        best[mode+"_test_f1_micro"] = float(micro_line.split()[-1])
                        best[mode+"_test_pr_micro"] = float(micro_line.split()[2])
                        best[mode+"_test_re_micro"] = float(micro_line.split()[4])
                      # macro
                        best[mode+"_test_f1_macro"] = float(macro_line.split()[-1])
                        best[mode+"_test_pr_macro"] = float(macro_line.split()[2])
                        best[mode+"_test_re_macro"] = float(macro_line.split()[4])

                except StopIteration:
                    break


      # APPEND SCORES for this log file
        model = ",".join([log_file.parts[-4], log_file.parts[-2]]).ljust(20)


        if "goldspans_eval_cnt" in best:
            checkpoint_file = log_file.parents[0]/"checkpoints/model-{}.meta".format(best["goldspans_eval_cnt"])
            checkpoint_text = ("yes" if checkpoint_file.exists() else "no")
          # micro
            if args.order_by_test_set:
                goldspans_micro.append((best["goldspans_test_f1_micro"], best["goldspans_dev_f1_micro"], model,
                                best["goldspans_eval_cnt"], checkpoint_text))
            else:
                goldspans_micro.append((best["goldspans_dev_f1_micro"],  best["goldspans_test_f1_micro"], model,
                                best["goldspans_eval_cnt"], checkpoint_text))
          # macro
            if args.order_by_test_set:
                goldspans_macro.append((best["goldspans_test_f1_macro"], best["goldspans_dev_f1_macro"], model,
                                best["goldspans_eval_cnt"], checkpoint_text))
            else:
                goldspans_macro.append((best["goldspans_dev_f1_macro"],  best["goldspans_test_f1_macro"], model,
                                best["goldspans_eval_cnt"], checkpoint_text))


        if "allspans_eval_cnt" in best:
            checkpoint_file = log_file.parents[0]/"checkpoints/model-{}.meta".format(best["allspans_eval_cnt"])
            checkpoint_text = ("yes" if checkpoint_file.exists() else "no")
          # micro
            if args.order_by_test_set:
                allspans_micro.append((best["allspans_test_f1_micro"], best["allspans_dev_f1_micro"], model,
                                best["allspans_eval_cnt"], checkpoint_text))
            else:
                allspans_micro.append((best["allspans_dev_f1_micro"],  best["allspans_test_f1_micro"], model,
                                best["allspans_eval_cnt"], checkpoint_text))
          # macro
            if args.order_by_test_set:
                allspans_macro.append((best["allspans_test_f1_macro"], best["allspans_dev_f1_macro"], model,
                                best["allspans_eval_cnt"], checkpoint_text))
            else:
                allspans_macro.append((best["allspans_dev_f1_macro"],  best["allspans_test_f1_macro"], model,
                                best["allspans_eval_cnt"], checkpoint_text))


def process_folder(args, goldspans_micro, allspans_micro, goldspans_macro, allspans_macro, input_folder):
    log_files = [f for f in input_folder.glob('**/log.txt')]
    for log_file in log_files:
        if file_is_used(log_file):
            print("File is being used by another process. Skip it.", log_file)
        else:
            pass
            process_experiment(args, goldspans_micro, allspans_micro, goldspans_macro, allspans_macro, log_file)


def main(args):
    """
    Print best results from log.txt files
    Best result is chosen by micro score on args.dev_set.
    To function properly
    THE RESULTS FROM dev HAVE TO APPEAR BEFORE test IN THE log.txt FILES
    """
    print("input_folder = "+str(args.input_folder)+"\n")

    goldspans_micro = []
    allspans_micro = []
    goldspans_macro = []
    allspans_macro = []

    if args.input_folder:
      # PROCESS TRAININGS FOLDER
        process_folder(args, goldspans_micro, allspans_micro, goldspans_macro, allspans_macro, args.input_folder)

      # SORT AND PRINT
        goldspans_micro = sorted(goldspans_micro, reverse=True)
        allspans_micro = sorted(allspans_micro, reverse=True)
        goldspans_macro = sorted(goldspans_macro, reverse=True)
        allspans_macro = sorted(allspans_macro, reverse=True)

        if goldspans_micro!=[] or goldspans_macro != []:
            print("\nGoldspans, micro:")
            if args.order_by_test_set:
                print("test_f1  dev_f1    model                epoch  checkpoint")
            else:
                print("dev_f1  test_f1    model                epoch  checkpoint")
            for t in goldspans_micro:
                print('\t'.join(map(str, t)))
            print("\nGoldspans,macro:")
            if args.order_by_test_set:
                print("test_f1  dev_f1    model                epoch  checkpoint")
            else:
                print("dev_f1  test_f1    model                epoch  checkpoint")
            for t in goldspans_macro:
                print('\t'.join(map(str, t)))

        if allspans_micro!=[] or allspans_macro != []:
            print("\nAllspans, micro:")
            if args.order_by_test_set:
                print("test_f1  dev_f1    model                epoch  checkpoint")
            else:
                print("dev_f1  test_f1    model                epoch  checkpoint")
            for t in allspans_micro:
                print('\t'.join(map(str, t)))
            print("\nAllspans, macro:")
            if args.order_by_test_set:
                print("test_f1  dev_f1    model                epoch  checkpoint")
            else:
                print("dev_f1  test_f1    model                epoch  checkpoint")
            for t in allspans_macro:
                print('\t'.join(map(str, t)))



def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", default="../data/experiments")
    parser.add_argument("--max_micro_or_macro", default='micro')
    #THE RESULTS FROM dev HAVE TO APPEAR BEFORE test IN THE log.txt FILES
    parser.add_argument("--dev_set", default="aida_dev")
    parser.add_argument("--test_set", default="aida_test")
    parser.add_argument("--order_by_test_set", dest="order_by_test_set", action="store_true")
    parser.add_argument("--no_order_by_test_set", dest="order_by_test_set", action="store_false")
    parser.set_defaults(order_by_test_set=False)

    args = parser.parse_args()

    args.input_folder=Path(args.input_folder)

    return args

if __name__ == "__main__":
    main(_parse_args())
