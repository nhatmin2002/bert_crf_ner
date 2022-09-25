#!/usr/bin/env python3

import argparse


def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="2", help="for distant debugging")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--checkpoint_path", type=str, default="./save_model")
    parser.add_argument("--pred_output_dir", type=str, default="./output_predict")

    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--do_train", type=bool, default=True)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--predict_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=64)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--crf_learning_rate", type=float, default=5e-5)

    parser.add_argument("--warmup_proportion", type=float, default=0.05)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    return parser
