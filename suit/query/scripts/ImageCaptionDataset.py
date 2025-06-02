import json
import os
from PIL import Image
import pandas as pd
import ast
from torch.utils.data import Dataset
from transformers import BatchEncoding


class ImageCaptionDataset(Dataset):
    def __init__(self) -> None:
        """
        Initializes the DataLoader.

        Args:
        - annotation_path: Path to the annotation file (JSON format).
        - image_folder: Path to the folder containing the images.
        - is_testset: Boolean indicating whether the dataset is a test set (i.e., no captions).
        """
        super().__init__()

    def __len__(self):
        """
        Returns the number of unique images in the dataset.

        Returns:
        - The number of unique images.
        """
        pass

    def get_img_path(self):
        """
        Returns the image path based on the image ID.

        Args:
        - image_id: The ID of the image.

        Returns:
        - The file path of the corresponding image.
        """
        pass

    def __getitem__(self, index):
        """
        Returns image and annotation information at the given index.

        Args:
        - index: The index of the sample to retrieve.

        Returns:
        - A dictionary with image and caption information.
        """
        return BatchEncoding()


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_train_dir_path,
        annotations_path,
        is_train,
        dataset_name,
        image_val_dir_path=None,
    ):
        self.image_train_dir_path = image_train_dir_path
        self.image_val_dir_path = image_val_dir_path
        self.annotations = []
        self.is_train = is_train
        self.dataset_name = dataset_name

        full_annotations = json.load(open(annotations_path))["images"]

        for i in range(len(full_annotations)):
            if self.is_train and full_annotations[i]["split"] != "train":
                continue
            elif not self.is_train and full_annotations[i]["split"] != "test":
                continue

            self.annotations.append(full_annotations[i])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if self.dataset_name == "coco":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
                if self.annotations[idx]["filepath"] == "train2014"
                else os.path.join(
                    self.image_val_dir_path, self.annotations[idx]["filename"]
                )
            )
        elif self.dataset_name == "flickr":
            image = Image.open(
                os.path.join(
                    self.image_train_dir_path, self.annotations[idx]["filename"]
                )
            )
        image.load()
        caption = self.annotations[idx]["sentences"][0]["raw"]
        return {
            "image": image,
            "caption": caption,
            "image_id": self.annotations[idx]["cocoid"]
            if self.dataset_name == "coco"
            else self.annotations[idx]["filename"].split(".")[0],
        }


class Flickr30kDataLoader(Dataset):
    def __init__(self, annotation_path, image_folder, is_testset=False):
        self.is_testset = is_testset
        self.image_folder = image_folder
        self.annotations = self.load_annotations(annotation_path)

    def load_annotations(self, annotation_path):
        """Load the annotations from the CSV file."""
        df = pd.read_csv(annotation_path)
        # df['raw'] = df['raw'].apply(self.parse_raw_column)

        df['raw'] = df['raw'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        return df

    def __len__(self):
        """Return the total number of annotations."""
        return len(self.annotations)

    def get_len_unique_image_ids(self):
        unique_filenames = self.annotations['filename'].unique()
        print(f'Total unique image IDs: {len(unique_filenames)}')
        return len(unique_filenames)

    def get_img_path(self, filename):
        """Return the image path based on the filename."""
        return os.path.join(self.image_folder, filename)

    def __getitem__(self, index):
        """Return image and annotation information."""
        if index >= len(self.annotations):
            raise IndexError("Index out of bounds.")

        annotation = self.annotations.iloc[index]
        img_path = self.get_img_path(annotation['filename'])
        # image = Image.open(img_path).convert("RGB")

        if self.is_testset:
            return {
                # "image": image,
                "image_id": annotation['filename'],
            }
        else:
            return {
                # "image": image,
                "image_id": annotation['filename'],
                "caption": annotation['raw'][0],
                "split": annotation['split']
            }


class COCODataLoader(ImageCaptionDataset):
    def __init__(self, annotation_path, image_folder=None, is_testset=False):
        self.is_testset = is_testset
        self.image_folder = image_folder
        self.annotations = self.load_annotations(annotation_path)

    def load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        # return data['annotations']
        annotations_by_image_id = {}
        for annotation in data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image_id:
                annotations_by_image_id[image_id] = []
            annotations_by_image_id[image_id].append(annotation['caption'])

        self.annotations_by_image_id = annotations_by_image_id
        return data['annotations']

    def __len__(self):
        """Counts the unique image IDs in the annotations."""
        image_ids = [annotation['image_id'] for annotation in self.annotations]
        unique_image_ids = set(image_ids)
        return len(unique_image_ids)
        # return len(self.annotations)

    def get_img_path(self, image_id):
        """Returns the image path based on the image ID."""
        image_id_str = "{:012d}".format(image_id)
        return os.path.join(self.image_folder, f'{image_id_str}.jpg')

    def __getitem__(self, index):
        """Returns image and annotation information."""
        unique_image_ids = list(self.annotations_by_image_id.keys())

        if index >= len(unique_image_ids):
            raise IndexError("Index out of bounds.")
        # if index >= len(self.annotations):
        #     raise IndexError("Index out of bounds.")

        # annotation = self.annotations[index]
        # image_id = annotation['image_id']
        image_id = unique_image_ids[index]
        img_path = self.get_img_path(image_id)
        image = Image.open(img_path).convert("RGB")

        if self.is_testset:
            return {
                "image": image,
                "image_id": image_id,
            }
        else:
            return {
                "image": image,
                "image_id": image_id,
                "caption": self.annotations_by_image_id[image_id]
            }


class CIRCODataLoader(ImageCaptionDataset):
    def __init__(self, annotation_path, image_folder=None, is_testset=False) -> None:
        self.is_testset = is_testset
        self.image_folder_path = image_folder
        self.annotations = self.load_annotations(annotation_path)

    def load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, query_id):
        """
        Returns the image path based on the query ID.
        """
        query = next(
            (item for item in self.annotations if item['id'] == query_id), None)
        if query:
            imageid = query['reference_img_id']
            imageid = "{:012d}".format(imageid)
            return os.path.join(self.image_folder_path, f"{imageid}.jpg")
        else:
            raise ValueError(f"No query found for id {query_id}")

    def __getitem__(self, index):
        """
        Returns a dictionary containing information 
        about the query at the given index.
        """

        if index >= len(self.annotations):
            raise IndexError("Index out of bounds.")

        query = self.annotations[index]
        if self.is_testset:
            return {
                "query_id": query['id'],
                "reference_img_id": query['reference_img_id'],
                "relative_caption": query.get("relative_caption", None),
                "shared_concept": query.get("shared_concept", None),
            }
        else:
            return {
                "query_id": query['id'],
                "reference_img_id": query['reference_img_id'],
                "target_img_id": query['target_img_id'],
                "relative_caption": query.get("relative_caption", None),
                "shared_concept": query.get("shared_concept", None),
                "gt_img_ids": query.get("gt_img_ids", []),
                "semantic_aspects": query.get("semantic_aspects", [])
            }

    def get_all_queries(self):
        """Returns a list of all queries."""
        queries = []
        for query in self.annotations:
            queries.append({
                "query_id": query['id'],
                "reference_img_id": query['reference_img_id'],
                "target_img_id": query['target_img_id'],
                "relative_caption": query.get("relative_caption", None),
                "shared_concept": query.get("shared_concept", None),
                "gt_img_ids": query.get("gt_img_ids", []),
                "semantic_aspects": query.get("semantic_aspects", [])
            })
        return queries


class JourneyDBDataLoader(ImageCaptionDataset):
    def __init__(self, annotation_path, image_folder=None, is_testset=False) -> None:
        self.is_testset = is_testset
        self.image_folder_path = image_folder
        self.annotations = self.load_annotations(annotation_path)

    def load_annotations(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = [json.loads(line) for line in f]
        return data

    def __len__(self):
        return len(self.annotations)

    def get_img_path(self, img_id):

        full_img_path = os.path.join(self.image_folder_path, f'{img_id}.jpg')

        return full_img_path

    def __getitem__(self, index):
        """
        Returns a dictionary containing information 
        about the query at the given index.
        """

        if index >= len(self.annotations):
            raise IndexError("Index out of bounds.")

        query = self.annotations[index]

        img_id = os.path.splitext(os.path.basename(query["img_path"]))[0]

        if self.is_testset:
            # For test set, return only part of the fields
            return {
                "image_id": img_id,
                "Task3_Style_QA": query["Task3"]["Style Relevant Questions and Answers"],
                "Task3_Content_QA": query["Task3"]["Content Relevant Questions and Answers"],
            }
        else:
            # For training/validation set, return all relevant fields
            return {
                "image_id": img_id,
                "prompt": query["prompt"],
                "Task1_Style": query["Task1"]["Style"],
                "Task1_Content": query["Task1"]["Content"],
                "Task1_Atmosphere": query["Task1"]["Atmosphere"],
                "Task1_Others": query["Task1"]["Others"],
                "Task2_Caption": query["Task2"]["Caption"],
                "Task3_Style_QA": query["Task3"]["Style Relevant Questions and Answers"],
                "Task3_Content_QA": query["Task3"]["Content Relevant Questions and Answers"],
            }


class GQADataset(ImageCaptionDataset):
    def __init__(self,  question_path, data_path) -> None:
        self.image_folder_path = data_path
        self.questions = self.load_questions(question_path)

    def __len__(self):
        return len(self.questions)

    def load_questions(self, questions_path):
        with open(questions_path, "r") as f:
            return json.load(f)

    def get_img_path(self, question_id):
        imageid = self.questions[f"{question_id}"]["imageId"]
        return os.path.join(self.image_folder_path, f"{imageid}.jpg")

    def __getitem__(self, index):
        question_pair = self.questions
        question_id = list(question_pair.keys())[index]

        question = question_pair[f"{question_id}"]["question"]

        img_path = self.get_img_path(question_id)
        image = Image.open(img_path)
        image.load()
        return {
            "image": image,
            "image_id": question_pair[f"{question_id}"]["imageId"],
            "question": question,
            "question_id": question_id,
            "answer": question_pair[f"{question_id}"]["answer"],
        }
