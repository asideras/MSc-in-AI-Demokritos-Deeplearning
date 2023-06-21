import torch
import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class InpaintedDataset(Dataset):
    def __init__(self, annotations_file, img_dir_inpainted,img_dir_original, transform=None, samples_type=None, num_training_samples=None,
                 num_validation_samples=None, num_testing_samples=None):        

        self.img_labels = pd.read_csv(annotations_file, dtype={'id': str})

        if samples_type == "Train":
            self.img_labels = self.img_labels.iloc[:num_training_samples, :]
        elif samples_type == "Validation":
            self.img_labels = self.img_labels.iloc[num_training_samples:num_training_samples+num_validation_samples, :]
        elif samples_type == "Test":
            self.img_labels = self.img_labels.iloc[num_training_samples+num_validation_samples:num_training_samples + num_validation_samples+num_testing_samples,:]

        self.img_dir_inpainted = img_dir_inpainted
        self.img_dir_original = img_dir_original

        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):

        fake_label = self.img_labels.iloc[idx,1]
        if fake_label :
            id = self.img_labels.iloc[idx, 0].lstrip('0')
            filename = f"{id}_second_out.jpg"
            img_path = os.path.join(self.img_dir_inpainted, filename)
        else:
            id = self.img_labels.iloc[idx,0]
            filename = f"Places365_val_{id}.jpg"
            img_path = os.path.join(self.img_dir_original, filename)

        image = read_image(img_path).float()

        if image.size(0) == 1:  # Check if image is one-channel
            image = image.repeat(3,1,1)


        if self.transform:
            image = self.transform(image)
        target = torch.tensor(self.img_labels.iloc[idx, 1:])

        return id, image, target.float()



#
# if __name__ == '__main__':
#     with open('../config.yaml', 'r') as file:
#         data = yaml.safe_load(file)
#
#     IMG_DIR = data['IMG_DIR']
#     ANNOTATIONS_FILE = data['ANNOTATIONS_FILE']
#     BATCH_SIZE = data['BATCH_SIZE']
#
#     test_df = pd.read_csv(ANNOTATIONS_FILE, dtype={'id': str})
#
#     print("Annotations_file size :",test_df.shape)
#
#     training_data = InpaintedDataset(annotations_file=ANNOTATIONS_FILE,
#                                 img_dir=IMG_DIR,
#                                 samples_type='Train')
#
#     train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=False)
#     print(f"Training set size: {len(training_data)}")
#
#     validation_data = InpaintedDataset(annotations_file=ANNOTATIONS_FILE,
#                                   img_dir=IMG_DIR,
#                                   samples_type='Validation')
#     validation_dataloader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
#     print(f"Validation set size: {len(validation_data)}")
#
#     test_data = InpaintedDataset(annotations_file=ANNOTATIONS_FILE,
#                             img_dir=IMG_DIR,
#                             samples_type='Test')
#
#     test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
#
#
#     print(test_df.head(2))
#
#     print(f"Train set size: {len(test_data)}")
#     train_iter = iter(train_dataloader)
#     train_id , train_image, train_target = next(train_iter)
#     print("first batch of training data loader:")
#     print(train_id[0])
#     print(test_df[test_df['id']=="0000000"+train_id[5]])
#     print(train_image.size())
#     print(train_target[5])
#
#
#     print(f"Validation set size: {len(validation_data)}")
#     validation_iter = iter(validation_dataloader)
#     validation_id , validation_image, validation_target = next(validation_iter)
#     print("first batch of validation data loader:")
#     print(validation_id[0])
#     print(validation_image.size())
#     print(validation_target.size())
#
#     print(f"Test set size: {len(test_data)}")
#     test_iter = iter(test_dataloader)
#     test_id , test_image, test_target = next(test_iter)
#     print("first batch of test data loader:")
#     print(test_id[0])
#     print(test_image.size())
#     print(test_target.size())

    # for id, inputs, targets in test_dataloader:
    #     print(id)

