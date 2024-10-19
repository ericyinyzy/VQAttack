from .base_dataset import BaseDataset


class VQAv2Dataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split
        # if split == "train":
        #     print('split_train')
        #     names = ["text_vqa_train"]
        # elif split == "val":
        #     print('split_val')
        #     names = ["text_vqa_train"]
        # elif split == "test":
        #     print('split_test')
        #     names = ["text_vqa_val","text_vqa_train"]

        if split == "train":
            print('split_train')
            names = ["vqav2_train", "vqav2_trainable_val"]
        elif split == "val":
            print('split_val')
            names = ["vqav2_trainable_val","vqav2_rest_val"]
        elif split == "test":
            print('split_test')
            names = ["vqav2_rest_val","vqav2_trainable_val"]  # vqav2_test-dev for test-dev
        # print(self.tokenizer)
        # exit()
        super().__init__(
            *args,
            **kwargs,
            names=names,
            text_column_name="questions",
            remove_duplicate=False,
        )

    def __getitem__(self, index):
        # print(self.tokenizer)#selff.test_dataset)
        # exit()
        image_tensor = self.get_image(index)["image"]
        text = self.get_text(index)["text"]
        # print(self)
        # exit()
        # print(self.tokenizer)
        # exit()
        # print(text)
        # exit()

        index, question_index = self.index_mapper[index]
        qid = self.table["question_id"][index][question_index].as_py()

        # if self.split != "test":
        answers = self.table["answers"][index][question_index].as_py()
        labels = self.table["answer_labels"][index][question_index].as_py()
        scores = self.table["answer_scores"][index][question_index].as_py()
        # else:
        #     answers = list()
        #     labels = list()
        #     scores = list()

        return {
            "image": image_tensor,
            "text": text,
            "vqa_answer": answers,
            "vqa_labels": labels,
            "vqa_scores": scores,
            "qid": qid,
        }
