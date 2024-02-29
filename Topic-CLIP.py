import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import requests
np.set_printoptions(threshold = np.inf)#numpy数组全部print

# url = 'https://huggingface.co/api/models/distilbert-base-uncased'
# requests.get(url, verify=False)
#print(timm.__version__)
#timm版本应为0.6.7，否则huggingface模型无法导入

class CFG:
    debug = False
    image_path = "all_image"
    captions_path = "."
    batch_size = 128
    num_workers = 0
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'resnet50'
    #model_name = 'vit_small_patch32_224'
    #model_name = 'vit_small_patch16_224_dino'
    image_embedding = 2048
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200
    pretrained = False # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0
    # image size
    size = 224
    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text



class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms, labels):
        """
        image_filenames and captions must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names
        """
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(list(captions), padding=True, truncation=True, max_length=CFG.max_length)
        self.transforms = transforms

        
        self.labels=labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(values[idx])
                for key, values in self.encoded_captions.items()}
        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]
        return item

    def __len__(self):
        return len(self.captions)


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """
    def __init__(self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
        for p in self.model.parameters():
            p.requires_grad = trainable
        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]


class ProjectionHead(nn.Module):
    def __init__(self,embedding_dim,projection_dim=CFG.projection_dim,dropout=CFG.dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

        
class CLIPModel(nn.Module):
    def __init__(self,temperature=CFG.temperature,image_embedding=CFG.image_embedding,text_embedding=CFG.text_embedding):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)
        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax((images_similarity + texts_similarity) / 2 * self.temperature, dim=-1)
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (images_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def get_transforms():
    return A.Compose([A.Resize(CFG.size, CFG.size, always_apply=True),
                      A.Normalize(max_pixel_value=255.0, always_apply=True)])


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


def make_train_valid_dfs():
    #dataframe = pd.read_csv(f"{CFG.captions_path}/2014_LabelTest.csv")
    dataframe = pd.read_csv(f"{CFG.captions_path}/2013_TextAndTopic.csv")
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    #np.random.seed(42)
    valid_ids = np.random.choice(image_ids, size=int(0.2 * len(image_ids)), replace=False)
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms()
    dataset = CLIPDataset(dataframe["image"].values,dataframe["caption"].values,tokenizer=tokenizer,transforms=transforms,labels=dataframe["label"].values)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=CFG.batch_size,num_workers=CFG.num_workers,shuffle=True if mode == "train" else False)
    return dataloader


def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total = len(train_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == "batch":
            lr_scheduler.step()
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def valid_epoch(model, valid_loader):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    for batch in tqdm_object:
        batch = {k: v.cuda() for k, v in batch.items() if k != "caption"}
        loss = model(batch)
        count = batch["image"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    return loss_meter


def get_image_embeddings(valid_df, model):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return torch.cat(valid_image_embeddings)


def get_text_embeddings(valid_df, model):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    valid_text_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            text_features = model.text_encoder(input_ids=batch["input_ids"].to(CFG.device), attention_mask=batch["attention_mask"].to(CFG.device))
            text_embeddings = model.text_projection(text_features)
            valid_text_embeddings.append(text_embeddings)
    return torch.cat(valid_text_embeddings)

def find_text_matches(model, image_filename, text_embeddings,captions,n=5):
    batch={}
    image = cv2.imread(f"{CFG.image_path}/{image_filename}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image=A.Compose([A.Resize(CFG.size, CFG.size, always_apply=True),
                     A.Normalize(max_pixel_value=255.0, always_apply=True),])(image=image)["image"]
    image = np.expand_dims(image, 0)
    batch["image"] = torch.tensor(image).permute(0, 3, 1, 2).float()
    batch["image"]=torch.tensor(batch["image"]).to(CFG.device)
    with torch.no_grad():
        image_features = model.image_encoder(batch["image"])
        image_embeddings = model.image_projection(image_features)        
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = image_embeddings_n @ text_embeddings_n.T 
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 3)
    matches = [captions[idx] for idx in indices[::3]]
    for i in range(n):
        print(matches[i])

def find_image_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    print(encoded_query.items())
    batch = {key: torch.tensor(values).to(CFG.device)
             for key, values in encoded_query.items()}
    with torch.no_grad():
        text_features = model.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        text_embeddings = model.text_projection(text_features)
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    values, indices = torch.topk(dot_similarity.squeeze(0), n*5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    plt.show()

def image_predict(model, image_embeddings, test_dataframe):
    TP=0
    TN=0
    FP=0
    FN=0
    text_embeddings = get_text_embeddings(test_dataframe, model)
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    labels_predict=np.zeros(dot_similarity.shape[1])
    for i in range(dot_similarity.shape[1]):
        labels_predict[i]=dot_similarity[0][i]-dot_similarity[1][i]
        if labels_predict[i]>0:
            labels_predict[i]=0
        else:
            labels_predict[i]=1
    for i in range(dot_similarity.shape[1]):
        if labels_predict[i]==1 and valid_df["label"][i]==1:
            TP=TP+1
        if labels_predict[i]==0 and valid_df["label"][i]==0:
            TN=TN+1
        if labels_predict[i]==1 and valid_df["label"][i]==0:
            FP=FP+1
        if labels_predict[i]==0 and valid_df["label"][i]==1:
            FN=FN+1

    Accuracy=(TP+TN)/(TP+FP+TN+FN)
    if TP==0 and FP==0:
        Precision=0
    else:
        Precision=TP/(TP+FP)
    Recall=TP/(TP+FN)
    if Precision==0 or Recall==0:
        F1Score=0
    else:
        F1Score=(2*Precision*Recall)/(Precision+Recall)
    return 100*Accuracy,100*Precision,100*Recall,100*F1Score,TP,TN,FP,FN

def train():
    train_df, valid_df = make_train_valid_dfs()
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    train_loader = build_loaders(train_df, tokenizer, mode="train")
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    model = CLIPModel()
    params = [{"params": model.image_encoder.parameters(), "lr": CFG.image_encoder_lr},
              {"params": model.text_encoder.parameters(), "lr": CFG.text_encoder_lr},
              {"params": itertools.chain(model.image_projection.parameters(),model.text_projection.parameters()), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}]
    #model = nn.DataParallel(model,device_ids=[0,1])
    model = model.cuda()
    optimizer = torch.optim.AdamW(params, weight_decay = 0.)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", patience = CFG.patience, factor = CFG.factor)
    step = "epoch"
    best_loss = float('inf')
    for epoch in range(CFG.epochs):
        print(f"Epoch: {epoch + 1}")
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        model.eval()
        with torch.no_grad():
            valid_loss = valid_epoch(model, valid_loader)
        if valid_loss.avg < best_loss:
            best_loss = valid_loss.avg
            torch.save(model.state_dict(), "ViTsmall16Pretrain.pt")
            print("Saved Best Model!")
        lr_scheduler.step(valid_loss.avg)

#数据读取及转换保存（可优化）
#df = pd.read_csv("test3.csv", delimiter="|")  
#df.columns = ['image', 'caption_number', 'caption'] 
#df['caption'] = df['caption'].astype(str).str.lstrip()
#df['caption_number'] = df['caption_number'].astype(str).str.lstrip()
#ids = [id_ for id_ in range(len(df) // 5) for i in range(5)]
#df['id'] = ids
#df.to_csv("captions.csv", index=False)
#df.head()

#train()
Accuracy= np.zeros(50)
Precision= np.zeros(50)
Recall= np.zeros(50)
F1Score= np.zeros(50) 

model = CLIPModel().to(CFG.device)
model.load_state_dict(torch.load("best1030.pt", map_location=CFG.device))
model.eval()



#dataframe = pd.read_csv(f"{CFG.captions_path}/text_caption.csv")
#image_embeddings = get_image_embeddings(dataframe, model)
#text_embeddings = get_text_embeddings(dataframe, model)
#find_text_matches(model,image_filename = "1663.jpg",text_embeddings = text_embeddings,captions=dataframe['caption'].values,n = 3)
#find_image_matches(model,image_embeddings = image_embeddings,query = "Depression",image_filenames=valid_df['image'].values,n = 9)
for i in tqdm(range(50)):
    _, valid_df = make_train_valid_dfs()
    image_embeddings = get_image_embeddings(valid_df, model)
    text_embeddings = get_text_embeddings(valid_df, model)
    df_test = pd.DataFrame(columns=['image','caption_number','caption','id','label'],data=[['test.jpg','0','normal','0','0'],['test.jpg','1','depress','1','1']])
    Accuracy[i],Precision[i],Recall[i],F1Score[i]=image_predict(model,image_embeddings = image_embeddings,test_dataframe = df_test)


print(Accuracy)
print(Precision)
print(Recall)
print(F1Score)