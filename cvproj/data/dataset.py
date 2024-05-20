from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import torchvision.transforms.functional as F
import json

from cvproj.data.process import canny_from_pil


class SketchyDataset(Dataset):
    def __init__(self, tokenizer, dataset_folder: str = "data/", split: str = "train"):
        super().__init__()

        self.input_folder = os.path.join(dataset_folder, split)
        captions = os.path.join(self.input_folder, "captions.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)

        self.img_names = list(self.captions.keys())
        self.T = transforms.Compose(
            [
                transforms.Resize((286, 286), interpolation=Image.LANCZOS),
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        output_img = Image.open(os.path.join(self.input_folder, img_name))
        input_img = canny_from_pil(
            output_img, low_threshold=300, high_threshold=350)
        caption = self.captions[img_name]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)

        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }

    def __len__(self):
        return len(self.captions)


class PixelDataset(Dataset):
    def __init__(self, tokenizer, split: str = "train"):
        super().__init__()
        dataset = load_dataset("m1guelpf/nouns")
        self.dataset = dataset[split]
        self.T = transforms.Compose(
            [
                transforms.Resize((286, 286), interpolation=Image.LANCZOS),
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # idx = idx % 10
        input_img = canny_from_pil(self.dataset[idx]["image"])
        output_img = self.dataset[idx]["image"]
        caption = self.dataset[idx]["text"]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)

        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }

    def __len__(self):
        # return 10
        return len(self.dataset)


class PokemonDataset(Dataset):
    def __init__(self, tokenizer, split: str = "train"):
        super().__init__()
        dataset = load_dataset("diffusers/pokemon-gpt4-captions")
        self.dataset = dataset[split]
        self.T = transforms.Compose(
            [
                transforms.Resize((286, 286), interpolation=Image.LANCZOS),
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # idx = idx % 10
        input_img = canny_from_pil(self.dataset[idx]["image"])
        output_img = self.dataset[idx]["image"]
        caption = self.dataset[idx]["text"]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)

        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }

    def __len__(self):
        # return 10
        return len(self.dataset)


class PairedDataset(Dataset):
    def __init__(self, dataset_folder, split, image_prep, tokenizer):
        super().__init__()
        if split == "train":
            self.input_folder = os.path.join(dataset_folder, "train_A")
            self.output_folder = os.path.join(dataset_folder, "train_B")
            captions = os.path.join(dataset_folder, "train_prompts.json")
        elif split == "test":
            self.input_folder = os.path.join(dataset_folder, "test_A")
            self.output_folder = os.path.join(dataset_folder, "test_B")
            captions = os.path.join(dataset_folder, "test_prompts.json")
        with open(captions, "r") as f:
            self.captions = json.load(f)
        self.img_names = list(self.captions.keys())
        self.T = transforms.Compose(
            [
                transforms.Resize((286, 286), interpolation=Image.LANCZOS),
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
            ]
        )
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        input_img = Image.open(os.path.join(self.input_folder, img_name))
        output_img = Image.open(os.path.join(self.output_folder, img_name))
        caption = self.captions[img_name]

        # input images scaled to 0,1
        img_t = self.T(input_img)
        img_t = F.to_tensor(img_t)
        # output images scaled to -1,1
        output_t = self.T(output_img)
        output_t = F.to_tensor(output_t)
        output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids

        return {
            "output_pixel_values": output_t,
            "conditioning_pixel_values": img_t,
            "caption": caption,
            "input_ids": input_ids,
        }

    def __len__(self):
        return len(self.captions)
