from PIL import Image
import numpy as np
import pandas as pd
import os
import argparse
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
import sklearn.metrics
import pickle

np.set_printoptions(precision=3)

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasetdir", default='Data/knowit_data/Frames', type=str)
    parser.add_argument("--resnetdir", default='Data/Video/Resnet50', type=str)
    parser.add_argument('--datafile', default='Data/knowit_data/knwoit_data_train.csv', type=str)
    parser.add_argument('--found_episode_file', default='Data/KnowledgeBase/retrieved_episode_from_scenes_train.csv', type=str)
    parser.add_argument("--bs", default=128, type=int)
    args, unknown = parser.parse_known_args()
    return args


def save_obj(obj, filename, verbose=True):
    f = open(filename, 'wb')
    pickle.dump(obj, f)
    f.close()
    if verbose:
        print("Saved object to %s." % filename)


def load_obj(filename, verbose=True):
    f = open(filename, 'rb')
    obj = pickle.load(f)
    f.close()
    if verbose:
        print("Load object from %s." % filename)
    return obj


def compute_frames_embeddings(args, frameslist, outfile):

    numframes = len(frameslist)

    # Prepare resnet model
    imgtransforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],
                             std=[0.229, 0.224, 0.225])
    ])
    resnet = models.resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.cuda()
    resnet.eval()

    # Load frames
    all_images = np.zeros((numframes, 3, 224, 224))
    for idx, file in enumerate(frameslist):
        image = Image.open(file).convert('RGB')
        image = imgtransforms(image)
        all_images[idx, :, :, :] = image
    all_images = torch.tensor(all_images, dtype=torch.float)
    eval_data = TensorDataset(all_images)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.bs)

    # Compute (pretrained) resnet encoding for each frame
    all_encodings = np.zeros((numframes, 2048))
    firstidx = 0
    for batch_idx, inputs in enumerate(eval_dataloader):
        frame = inputs[0].to('cuda')
        frame_encoding = resnet(frame)
        frame_encoding = torch.squeeze(frame_encoding)
        frame_encoding = frame_encoding.data.cpu().numpy()
        numSamplesBatch =  frame_encoding.shape[0]
        all_encodings[firstidx:firstidx+numSamplesBatch,:] = frame_encoding
        firstidx = firstidx + numSamplesBatch

    save_obj(all_encodings, outfile, verbose=False)


def get_episode_from_clip(args, clipframeslist, episodeslist):

    numframesclip = len(clipframeslist)

    # Prepare resnet model
    imgtransforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],
                             std=[0.229, 0.224, 0.225])
    ])
    resnet = models.resnet50(pretrained=True)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet.cuda()
    resnet.eval()

    # Load frames
    all_images = np.zeros((numframesclip, 3, 224, 224))
    for idx, file in enumerate(clipframeslist):
        image = Image.open(file).convert('RGB')
        image = imgtransforms(image)
        all_images[idx, :, :, :] = image
    all_images = torch.tensor(all_images, dtype=torch.float)
    eval_data = TensorDataset(all_images)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.bs)

    # Compute (pretrained) resnet encoding for each frame
    clip_encodings = np.zeros((numframesclip, 2048))
    firstidx = 0
    for batch_idx, inputs in enumerate(eval_dataloader):
        frame = inputs[0].to('cuda')
        frame_encoding = resnet(frame)
        frame_encoding = torch.squeeze(frame_encoding)
        frame_encoding = frame_encoding.data.cpu().numpy()
        numSamplesBatch =  frame_encoding.shape[0]
        clip_encodings[firstidx:firstidx+numSamplesBatch,:] = frame_encoding
        firstidx = firstidx + numSamplesBatch

    # Find best matching episode for each frame in the clip
    best_scores = np.zeros((numframesclip, len(episodeslist)))
    for idx, episode in enumerate(episodeslist):
        embsfile = os.path.join(args.resnetdir, episode + '.pckl')
        episode_encodings = load_obj(embsfile, verbose=False)
        similarities = sklearn.metrics.pairwise.cosine_similarity(clip_encodings, episode_encodings)
        best_scores[:,idx] = np.max(similarities, axis=1)

    episode_scores = np.sum(best_scores, axis=0)
    idmax = np.argmax(episode_scores)
    scoremax = np.max(episode_scores) / numframesclip
    found_episode = episodeslist[idmax]

    return [found_episode, scoremax]


def dataset_process(args):

    # Get list of frames
    dfframes = pd.read_csv(os.path.join(args.datasetdir, 'list_frames.csv'), sep='\t')
    allepisodes = dfframes['frame_path'].str.split('/').str[0].unique()

    if not os.path.isdir(args.resnetdir):
        os.mkdir(args.resnetdir)

    for episode in tqdm(allepisodes, desc="Episodes"):

        outfile = os.path.join(args.resnetdir, episode + '.pckl')
        if not os.path.exists(outfile):
            frameslist = [os.path.join(args.datasetdir, row['frame_path']) for idx, row in dfframes.iterrows() if row['frame_path'].split('/')[0] == episode]
            compute_frames_embeddings(args, frameslist, outfile)


def eval_process(args):

    df = pd.read_csv(args.datafile, sep='\t')
    allscenes = df['scene'].unique()
    num_samples = len(allscenes)

    # Get list of frames
    dfframes = pd.read_csv(os.path.join(args.datasetdir, 'list_frames.csv'), sep='\t')
    allepisodes = dfframes['frame_path'].str.split('/').str[0].unique()

    # Output file
    with open(args.found_episode_file, 'w') as fout:
        fout.write('Scene\tFound Episode\n')

        acc = 0
        for scene in tqdm(allscenes, desc="Clips"):
            [episode, _, start, end] = scene.split('_')
            frameslist = [os.path.join(args.datasetdir, episode, 'frame_%04d.jpeg' % idframe) for idframe in list(range(int(start), int(end)))]
            [found_episode, scoremax] = get_episode_from_clip(args, frameslist, allepisodes)
            if found_episode == episode:
                acc += 1

            # save for future use
            fout.write('%s\t%s\n' % (scene, found_episode))

    acc = acc / num_samples
    print('Episode retrieval accuracy is %.03f' % acc)


if __name__ == "__main__":

    args = get_params()
    dataset_process(args)
    eval_process(args)

