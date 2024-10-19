import json
import pandas as pd
import pyarrow as pa
import random
import os

from tqdm import tqdm
from glob import glob
from collections import defaultdict, Counter
from glossary import normalize_word


def get_score(occurences):
    if occurences == 0:
        return 0.0
    elif occurences == 1:
        return 0.3
    elif occurences == 2:
        return 0.6
    elif occurences == 3:
        return 0.9
    else:
        return 1.0


def path2rest(path, split, annotations):
    iid = str(path.split("/")[-1].strip('.jpg'))
    # print('path',path)

    with open(path, "rb") as fp:
        binary = fp.read()

    _annot = annotations[split][iid]
    _annot = list(_annot.items())
    qids, qas = [a[0] for a in _annot], [a[1] for a in _annot]
    # print(qids,qas)
    # exit()
    questions = [qa[0] for qa in qas]
    # print(questions)
    answers = [qa[1]['answers'] for qa in qas] if "test" not in split else list(list())
    answer_scores = (
        [qa[1]['scores'] for qa in qas] if "test" not in split else list(list())
    )
    answer_labels=[[0 for i in range(len(score))] for score in answer_scores]
    # answer_labels = (
    #     [a["labels"] for a in answers] if "test" not in split else list(list())
    # )
    # answer_scores = (
    #     [a["scores"] for a in answers] if "test" not in split else list(list())
    # )
    # answers = (
    #     [[label2ans[l] for l in al] for al in answer_labels]
    #     if "test" not in split
    #     else list(list())
    # )
    # print( questions, answers, answer_labels,answer_scores, iid, qids, split)
    # exit()

    return [binary, questions, answers, answer_labels,answer_scores, iid, qids, split]


def make_arrow(root, dataset_root='/data/ziyi/TextVQA_Arrow'):
    with open(f"{root}/TextVQA_0.5.1_train.json", "r") as fp:
        questions_train = json.load(fp)["data"]
    with open(f"{root}/TextVQA_0.5.1_val.json", "r") as fp:
        questions_val = json.load(fp)["data"]

    annotations = dict()

    for split, questions in zip(
        ["train", "val"],
        [
            questions_train,
            questions_val,
        ],
    ):
        _annot = defaultdict(dict)
        # print(_annot)
        # exit()
        for q in tqdm(questions):
            # print(type(q["question_id"]))
            # exit()
            _annot[q["image_id"]][q["question_id"]] = [q["question"]]
        # exit()

        annotations[split] = _annot

    # all_major_answers = list()
    #
    # for split, annots in zip(
    #     ["train", "val"], [questions_train, questions_val],
    # ):
    #     _annot = annotations[split]
    #     for q in tqdm(annots):
    #         all_major_answers.append(q["answers"])
    #
    # all_major_answers = [normalize_word(word) for word in tqdm(all_major_answers)]
    # counter = {k: v for k, v in Counter(all_major_answers).items() if v >= 9}
    # ans2label = {k: i for i, k in enumerate(counter.keys())}
    # label2ans = list(counter.keys())

    for split, annots in zip(
        ["train", "val"], [questions_train, questions_val],
    ):
        _annot = annotations[split]
        for q in tqdm(annots):
            # print(q)
            # exit()
            answers = q["answers"]
            answer_count = {}
            # answer_list
            for answer in answers:
                # answer_ = answer["answer"]
                answer_count[answer] = answer_count.get(answer, 0) + 1
            # print(answer_count)

            labels = []
            scores = []
            for answer in answer_count:
                # if answer not in ans2label:
                #     continue
                # labels.append(ans2label[answer])
                score = get_score(answer_count[answer])
                scores.append(score)
            # print(scores)
            # exit()

            _annot[q["image_id"]][q["question_id"]].append(
                {"answers": [k for k in answer_count.keys()], "scores": scores,}
            )
            # print(_annot[q["image_id"]][q["question_id"]])
            # exit()

    # for split in ["train", "val"]:
    #     filtered_annot = dict()
    #     for ik, iv in annotations[split].items():
    #         new_q = dict()
    #         for qk, qv in iv.items():
    #             if len(qv[1]["labels"]) != 0:
    #                 new_q[qk] = qv
    #         if len(new_q) != 0:
    #             filtered_annot[ik] = new_q
    #     annotations[split] = filtered_annot

    for split in [
        "train",
        "val",
    ]:
        annot = annotations[split]
        split_name = {
            "train": "train_images",
            "val": "train_images",
        }[split]
        # print(annot)
        # exit()
        paths = list(glob(f"{root}/{split_name}/*.jpg"))
        random.shuffle(paths)
        # print(paths)
        annot_paths = [
            path
            for path in paths
            if str(path.split("/")[-1].strip(".jpg")) in annot
        ]

        if len(paths) == len(annot_paths):
            print("all images have caption annotations")
        else:
            print("not all images have caption annotations")
        print(
            len(paths), len(annot_paths), len(annot),
        )
        # exit()

        bs = [
            path2rest(path, split, annotations) for path in tqdm(annot_paths)
        ]
        # exit()

        dataframe = pd.DataFrame(
            bs,
            columns=[
                "image",
                "questions",
                "answers",
                "answer_labels",
                "answer_scores",
                "image_id",
                "question_id",
                "split",
            ],
        )

        table = pa.Table.from_pandas(dataframe)

        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(f"{dataset_root}/text_vqa_{split}.arrow", "wb") as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)

    # table = pa.ipc.RecordBatchFileReader(
    #     pa.memory_map(f"{dataset_root}/vqav2_val.arrow", "r")
    # ).read_all()
    #
    # pdtable = table.to_pandas()
    #
    # df1 = pdtable[:-1000]
    # df2 = pdtable[-1000:]
    #
    # df1 = pa.Table.from_pandas(df1)
    # df2 = pa.Table.from_pandas(df2)
    #
    # with pa.OSFile(f"{dataset_root}/vqav2_trainable_val.arrow", "wb") as sink:
    #     with pa.RecordBatchFileWriter(sink, df1.schema) as writer:
    #         writer.write_table(df1)
    #
    # with pa.OSFile(f"{dataset_root}/vqav2_rest_val.arrow", "wb") as sink:
    #     with pa.RecordBatchFileWriter(sink, df2.schema) as writer:
    #         writer.write_table(df2)
make_arrow(root='/data/ziyi/TextVQA')