import argparse
import itertools
import random
import numpy as np
from config import Reader, Config, ContextEmb, lr_decay, evaluate_batch_insts, get_optimizer, write_results, batching_list_instances
import time
from model import NNCRF, TransformersCRF
import torch
from typing import List, Tuple
from common import Instance
from termcolor import colored
import os
from config.utils import load_elmo_vec
from config.transformers_util import tokenize_instance, get_huggingface_optimizer_and_scheduler
from config import context_models, get_metric
import pickle
import tarfile
from tqdm import tqdm
from collections import Counter
import torch.nn as nn


def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=False,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="conll2003_sample")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="adamw", help="This would be useless if you are working with transformers package")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="usually we use 0.01 for sgd but 2e-5 working with bert/roberta")
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=30, help="default batch size is 10 (works well for normal neural crf), here default 30 for bert-based crf")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--max_no_incre', type=int, default=30, help="early stop when there is n epoch not increasing on dev")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm, if <=0, means no clipping, usually we don't use clipping for normal neural ncrf")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english_model", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=0, help="hidden size of the LSTM, usually we set to 200 for LSTM-CRF")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--static_context_emb', type=str, default="none", choices=["none", "elmo"],
                        help="static contextual word embedding, our old ways to incorporate ELMo and BERT.")

    parser.add_argument('--embedder_type', type=str, default="bert-base-cased",
                        choices=["normal"] + list(context_models.keys()),
                        help="normal means word embedding + char, otherwise you can use 'bert-base-cased' and so on")
    parser.add_argument('--parallel_embedder', type=int, default=0,
                        choices=[0, 1],
                        help="use parallel training for those (BERT) models in the transformers. Parallel on GPUs")

    parser.add_argument('--num_outer_iterations', type= int , default= 2, help="Number of outer iterations for cross validation")



    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

def train_one(config: Config, train_insts: List[Instance], dev_insts: List[Instance], model_name: str, test_insts: List[Instance] = None,
              config_name: str = None, result_filename: str = None):
    train_batch_size = len(train_insts) // config.batch_size + 1
    epoch = config.num_epochs
    print(
        colored(f"[Model Info]: Working with transformers package from huggingface with {config.embedder_type}", 'red'))
    print(colored(f"[Optimizer Info]: You should be aware that you are using the optimizer from huggingface.", 'red'))
    print(colored(f"[Optimizer Info]: Change the optimier in transformers_util.py if you want to make some modifications.", 'red'))
    model = TransformersCRF(config)
    optimizer, scheduler = get_huggingface_optimizer_and_scheduler(config, model, num_training_steps=train_batch_size * epoch,
                                                                   weight_decay=0.0,
                                                                   eps = 1e-8,
                                                                   warmup_step=0)
    print(colored(f"[Optimizer Info] Modify the optimizer info as you need.", 'red'))
    print(optimizer)

    model.to(config.device)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    # if os.path.exists("model_files/" + model_folder):
    #     raise FileExistsError(
    #         f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
    #         f"to avoid override.")
    model_path = f"model_files/{model_folder}/lstm_crf.m"
    config_path = f"model_files/{model_folder}/config.conf"
    res_path = f"{res_folder}/{model_folder}.results"
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    os.makedirs(f"model_files/{model_folder}", exist_ok= True) ## create model files. not raise error if exist
    os.makedirs(res_folder, exist_ok=True)
    no_incre_dev = 0
    print(colored(f"[Train Info] Start training, you have set to stop if performace not increase for {config.max_no_incre} epochs",'red'))

    train_batches = batching_list_instances(config, train_insts)
    for i in tqdm(range(1, epoch + 1), desc="Epoch"):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in tqdm(np.random.permutation(train_batch_size), desc="--training batch", total=train_batch_size):
            model.train()
            loss = model(**train_batches[index])
            epoch_loss += loss.item()
            loss.backward()
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            model.zero_grad()
            if scheduler is not None:
                scheduler.step()
        end_time = time.time()
        print("Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, "dev", dev_insts)
        if test_insts:
            test_metrics = evaluate_model(config, model, "test", test_insts)
        if dev_metrics[2] > best_dev[0]:
            print("saving the best model...")
            no_incre_dev = 0
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            if test_insts:
                best_test[0] = test_metrics[2]
            best_test[1] = i
            torch.save(model.state_dict(), model_name)
            # Save the corresponding config as well.
            f = open(config_path, 'wb')
            pickle.dump(config, f)
            f.close()
            if test_insts:
                write_results(res_path, test_insts)
        else:
            no_incre_dev += 1
        model.zero_grad()
        if no_incre_dev >= config.max_no_incre:
            print("early stop because there are %d epochs not increasing f1 on dev"%no_incre_dev)
            break

    print("Archiving the best Model...")
    with tarfile.open(f"model_files/{model_folder}/{model_folder}.tar.gz", "w:gz") as tar:
        tar.add(f"model_files/{model_folder}", arcname=os.path.basename(model_folder))

    print("Finished archiving the models")

    print("The best dev: %.2f" % (best_dev[0]))
    if test_insts:
        print("The corresponding test: %.2f" % (best_test[0]))
    print("Final testing.")
    model.load_state_dict(torch.load(model_name))
    if test_insts:
        model.eval()
        evaluate_model(config, model, "test", test_insts)
        write_results(res_path, test_insts)

def train_model_on_splitted_train(config: Config, train_insts: List[List[Instance]], dev_insts: List[Instance]):
    model_folder = config.model_folder
    model_names = []  # model names for each fold
    for fold_id, folded_train_insts in enumerate(train_insts):
        print(f"[Training Info] Training fold {fold_id}.")
        model_name = model_folder + f"/lstm_crf_{fold_id}.m"
        model_names.append(model_name)
        train_one(config=config, train_insts=train_insts[fold_id],
                  dev_insts=dev_insts, model_name=model_name)
    return model_names

def predict_with_constraints(config: Config, model: TransformersCRF, fold_batches: List[Tuple], folded_insts:List[Instance]):
    batch_id = 0
    batch_size = config.batch_size
    model.eval()
    for batch in fold_batches:
        one_batch_insts = folded_insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        word_seq_lens = batch[1].cpu().numpy()
        with torch.no_grad():
            batch_max_scores, batch_max_ids = model.decode(batch)
        batch_max_ids = batch_max_ids.cpu().numpy()
        for idx in range(len(batch_max_ids)):
            length = word_seq_lens[idx]
            prediction = batch_max_ids[idx][:length].tolist()
            prediction = prediction[::-1]
            one_batch_insts[idx].output_ids = prediction

def update_train_insts(config: Config, train_insts:  List[List[Instance]], model_names):
    # assign hard prediction to other folds
    print("\n\n[Data Info] Assigning labels for the HARD approach")
    train_batches = [batching_list_instances(config, insts) for insts in train_insts]
    for fold_id, folded_train_insts in enumerate(train_insts):
        model = TransformersCRF(config)
        model_name = model_names[fold_id]
        model.load_state_dict(torch.load(model_name))
        predict_with_constraints(config=config, model=model,
                                 fold_batches=train_batches[1 - fold_id],
                                 folded_insts=train_insts[1 - fold_id])  ## set a new label id

    print("\n\n")
    return train_insts

def evaluate_on_test(config: Config, all_train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]):
    print("[Training Info] Training the final model")
    model_folder = config.model_folder
    res_folder = config.model_folder
    # めんどくさいので一緒にしちゃお
    model_name = model_folder + "/final_lstm_crf.m"
    config_name = model_folder + "/config.conf"
    res_name = res_folder + "/lstm_crf.results".format()
    model = train_one(config=config, train_insts=all_train_insts, dev_insts=dev_insts,
                      model_name=model_name, config_name=config_name, test_insts=test_insts,
                      result_filename=res_name)
    print("Archiving the best Model...")
    with tarfile.open(model_folder + "/" + model_folder + ".tar.gz", "w:gz") as tar:
        tar.add(model_folder, arcname=os.path.basename(model_folder))
    # print("The best dev: %.2f" % (best_dev[0]))
    # print("The corresponding test: %.2f" % (best_test[0]))
    # print("Final testing.")
    model.load_state_dict(torch.load(model_name))
    model.eval()
    evaluate_model(config, model, "test", test_insts)
    write_results(res_name, test_insts)

def train_model(config: Config, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]):
    ### Data Processing Info
    epoch = config.num_epochs
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    # only hard model of (Jie et al. 2019)
    for inst in dev_insts:
        inst.is_prediction = [False] * len(inst.input)
        for pos, label in enumerate(inst.output):
            if label == config.O:
                inst.is_prediction[pos] = True

    import math
    num_insts_in_fold = math.ceil(len(train_insts) / config.num_folds)
    train_instss = [train_insts[i * num_insts_in_fold: (i + 1) * num_insts_in_fold] for i in range(config.num_folds)]

    num_outer_iterations = config.num_outer_iterations
    for iter in range(num_outer_iterations):
        print(f"[Training Info] Running for {iter}th large iterations.")
        model_names = train_model_on_splitted_train(config, train_instss, dev_insts)
        train_instss = update_train_insts(config, train_instss, model_names)
        all_train_insts = list(itertools.chain.from_iterable(train_instss))
        evaluate_on_test(config, all_train_insts, dev_insts, test_insts)




def evaluate_model(config: Config, model: TransformersCRF, name: str, insts: List[Instance], print_each_type_metric: bool = False):
    ## evaluation
    batch_insts_ids = batching_list_instances(config, insts)
    p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
    batch_id = 0
    batch_size = config.batch_size
    with torch.no_grad():
        for batch in batch_insts_ids:
            one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
            batch_max_scores, batch_max_ids = model.decode(**batch)
            batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts, batch_max_ids, batch["labels"], batch["word_seq_lens"], config.idx2labels)
            p_dict += batch_p
            total_predict_dict += batch_predict
            total_entity_dict += batch_total
            batch_id += 1
    if print_each_type_metric:
        for key in total_entity_dict:
            precision_key, recall_key, fscore_key = get_metric(p_dict[key], total_entity_dict[key], total_predict_dict[key])
            print(f"[{key}] Prec.: {precision_key:.2f}, Rec.: {recall_key:.2f}, F1: {fscore_key:.2f}")

    total_p = sum(list(p_dict.values()))
    total_predict = sum(list(total_predict_dict.values()))
    total_entity = sum(list(total_entity_dict.values()))
    precision, recall, fscore = get_metric(total_p, total_entity, total_predict)
    print(colored(f"[{name} set Total] Prec.: {precision:.2f}, Rec.: {recall:.2f}, F1: {fscore:.2f}", 'blue'), flush=True)


    return [precision, recall, fscore]


def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    trains = reader.read_txt(conf.train_file, conf.train_num)
    devs = reader.read_txt(conf.dev_file, conf.dev_num)
    tests = reader.read_txt(conf.test_file, conf.test_num)

    if conf.static_context_emb != ContextEmb.none:
        print('Loading the static ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.static_context_emb.name + ".vec", trains)
        load_elmo_vec(conf.dev_file + "." + conf.static_context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.static_context_emb.name + ".vec", tests)

    conf.use_iobes(trains + devs + tests)
    conf.build_label_idx(trains + devs + tests)

    if conf.embedder_type == "normal":
        conf.build_word_idx(trains, devs, tests)
        conf.build_emb_table()

        conf.map_insts_ids(trains)
        conf.map_insts_ids(devs)
        conf.map_insts_ids(tests)
        print("[Data Info] num chars: " + str(conf.num_char))
        # print(str(conf.char2idx))
        print("[Data Info] num words: " + str(len(conf.word2idx)))
        # print(config.word2idx)
    else:
        """
        If we use the pretrained model from transformers
        we need to use the pretrained tokenizer
        """
        print(colored(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer", "red"))
        tokenize_instance(context_models[conf.embedder_type]["tokenizer"].from_pretrained(conf.embedder_type), trains + devs + tests, conf.label2idx)

    train_model(conf, trains, devs, tests)


if __name__ == "__main__":
    main()
