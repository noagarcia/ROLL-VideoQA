import os
import pandas as pd
import ast
import argparse
from tqdm import tqdm
import pickle
import torch as nn
import json
from Source import utils_graphs as ugraphs

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


HUMAN_LABELS = ['boy', 'girl', 'guy', 'lady', 'man', 'person', 'player', 'woman']


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='Data/', type=str)
    parser.add_argument("--dataset", default='knowit', type=str, help='knowit or tvqa')
    parser.add_argument('--output_file', default='scenes_descriptions.csv')
    parser.add_argument('--characters_file',default='knowit_character_recognition.tsv')
    parser.add_argument('--places_file', default='knowit_places_classification.csv')
    parser.add_argument('--actions_file', default='knowit_action_predictions.pkl')
    parser.add_argument('--objectrel_dir', default='knowit-vrd/')
    parser.add_argument('--actions_vocab_file', default='actions_charades_classes.txt')
    parser.add_argument('--actions_framelist_file', default='actions_framelist.csv')
    parser.add_argument('--objs_file', default='vg_objects.json')
    parser.add_argument('--preds_file', default='vg_predicates.json')
    parser.add_argument('--topk', type=int, default=100)
    args, unknown = parser.parse_known_args()
    return args


def replace_with_name(objbox, objlabel, faceboxes, facelabels, verbose = False):

    obj_l, obj_t, obj_r, obj_b = objbox
    newlabel = objlabel
    max_score = 0
    for (face_t, face_r, face_b, face_l), name in zip(faceboxes, facelabels):

        if name == 'unknown':
            continue

        (inter_l, inter_t, inter_r, inter_b) = ugraphs.boxes_intersect(obj_l, obj_t, obj_r, obj_b,
                                                                       face_l, face_t, face_r, face_b)
        area_inter = ugraphs.box_area(inter_l, inter_t, inter_r, inter_b)
        area_face = ugraphs.box_area(face_l, face_t, face_r, face_b)
        area_object = ugraphs.box_area(obj_l, obj_t, obj_r, obj_b)
        score_bbox_face = area_inter / area_face
        score_bbox_object = area_inter / area_object

        if score_bbox_face > 0.9 and score_bbox_face <= 1 and score_bbox_object > max_score:
            newlabel = name
            max_score = score_bbox_object

        if verbose:
            print('score_bbox_face: ' + str(score_bbox_face))
            print('score_bbox_object: ' + str(score_bbox_object))

    return newlabel


def get_all_scenes_dataset(dataset):

    if dataset == 'knowit':
        df1 = pd.read_csv('Data/knowit_data/knowit_data_train.csv', sep='\t')
        df2 = pd.read_csv('Data/knowit_data/knowit_data_val.csv', sep='\t')
        df3 = pd.read_csv('Data/knowit_data/knowit_data_test.csv', sep='\t')
        df = pd.concat([df1, df2, df3], sort=True)
        allscenes = df['scene'].unique().tolist()

    return allscenes


def load_visual_features(args):

    # characters
    df_characters = pd.read_csv(os.path.join(args.data_dir, args.characters_file), delimiter='\t')

    # places
    df_places = pd.read_csv(os.path.join(args.data_dir, args.places_file), delimiter='\t')

    # objects relations
    with open(os.path.join('Data/', args.objs_file)) as json_file:
        objs_dict = json.load(json_file)
    with open(os.path.join('Data/', args.preds_file)) as json_file:
        preds_dict = json.load(json_file)

    # actions
    pdactions = pd.read_csv(os.path.join('Data/', args.actions_vocab_file), header=None)[0].to_list()
    actions_vocab = [a[5:] for a in pdactions]
    action_preds, _ = pickle.load(open(os.path.join(args.data_dir, args.actions_file), 'rb'), encoding='latin1')
    action_scene_order = list(pd.read_csv(os.path.join('Data/', args.actions_framelist_file), sep=' ')['original_vido_id'].unique())

    return df_characters, df_places, action_scene_order, action_preds, actions_vocab, objs_dict, preds_dict


def get_characters_scene(df_characters, frameslist):
    allnames = []
    for idx, file in enumerate(frameslist):
        facesframe = df_characters[df_characters['frame_path'] == file].iloc[0]
        listnames = ast.literal_eval(facesframe['people'])
        allnames.extend(listnames)
    allnames = list(set(allnames))
    return allnames


def get_place_scene(df_places, frameslist):
    middleframe = frameslist[int(len(frameslist) / 2)]
    placesframe = df_places[df_places['Frame'] == middleframe].iloc[0]
    place_scene = placesframe['Location Accumulated']
    return place_scene


def get_mainaction_scene(action_scene_order, scenename, action_preds, actions_vocab):
    actionId = action_scene_order.index(scenename)
    predsSample = action_preds[actionId, :]
    scoresAction = nn.softmax(nn.from_numpy(predsSample), dim=0)
    [_, idactions_top] = scoresAction.topk(5)
    action_scene = actions_vocab[int(idactions_top[0])]
    return action_scene


def get_objrelations_scene(episode, frameslist, df_characters, objs_dict, preds_dict):

    # Load graph episode
    f = open(os.path.join(args.data_dir, args.objectrel_dir, episode + '.pkl'), 'rb')
    anns_episode = pickle.load(f)
    f.close()

    all_label_subj, all_label_obj, all_predicate, all_score, all_faces = [], [], [], [], []
    for idxFrame, file in enumerate(frameslist):

        # Get characters bounding boxes, objects and relations
        facesframe = df_characters[df_characters['frame_path'] == file].iloc[0]
        facelabels = ast.literal_eval(facesframe['people'])
        faceboxes = ast.literal_eval(facesframe['boxes'])
        anns_frame_all = [ann for ann in anns_episode if ann['image'] == file][0]
        anns_frame_top = ugraphs.select_top_triplets(anns_frame_all, args.topk)

        # For each triplet (object, relation, subject) assign labels and filter low score triplets
        for idxObj in list(range(len(anns_frame_top['det_scores_top']))):
            label_subj = objs_dict[anns_frame_top['det_labels_s_top'][idxObj]]
            label_obj = objs_dict[anns_frame_top['det_labels_o_top'][idxObj]]
            predicate = preds_dict[anns_frame_top['det_labels_p_top'][idxObj]]
            score_triplet = anns_frame_top['det_scores_top'][idxObj]
            bbox_subj = anns_frame_top['det_boxes_s_top'][idxObj]
            bbox_obj = anns_frame_top['det_boxes_o_top'][idxObj]

            if score_triplet < 0.1:
                continue

            # replace person bounding boxes by detected character
            if label_subj in HUMAN_LABELS:
                newlabel = replace_with_name(bbox_subj, label_subj, faceboxes, facelabels, verbose=False)
                label_subj = newlabel

            if label_obj in HUMAN_LABELS:
                newlabel = replace_with_name(bbox_obj, label_obj, faceboxes, facelabels, verbose=False)
                label_obj = newlabel

            all_label_subj.append(label_subj)
            all_label_obj.append(label_obj)
            all_predicate.append(predicate)
            all_score.append(score_triplet)
            all_faces.extend(facelabels)

    all_faces = list(set(all_faces))
    if 'unknown' in all_faces:
        all_faces.remove('unknown')

    # For each pair of sub-obj only one triplet (highest score)
    seen = set()
    all_triplets = sorted(zip(all_score, all_label_subj, all_predicate,all_label_obj), reverse=True)
    all_triplets = [(sub, pred, obj, sc) for sc, sub, pred, obj in all_triplets
              if not ((sub, obj) in seen or seen.add((sub, obj)) or seen.add((obj, sub)))]
    new_label_subj, new_predicate, new_label_obj, new_score = [], [], [], []
    if len(all_triplets) >= 1:
        new_label_subj, new_predicate, new_label_obj, new_score = zip(*all_triplets)
    return [new_label_subj, new_predicate, new_label_obj, new_score]


def create_description(faces_scene, place_scene, action_scene, objs_scene):

    # Characters
    if 'unknown' in faces_scene:
        faces_scene.remove('unknown')
    if len(faces_scene) < 1:
        faces_sentence = ''
    elif len(faces_scene) == 1:
        faces_sentence = faces_scene[0]
    else:
        faces_sentence = ', '.join(faces_scene[0:-1]) + ' and ' + faces_scene[-1]

    # Places
    if not place_scene == 'unk':
        places_sentence = ' at ' + place_scene
    else:
        places_sentence = '.'

    # Actions
    action_scene = action_scene[0].lower() + action_scene[1:]
    if len(faces_scene) > 1:
        actions_sentence = ' are ' + action_scene
    elif len(faces_scene) == 1:
        actions_sentence = ' is ' + action_scene
    else:
        actions_sentence = 'Someone is ' + action_scene

    # Objects sentences
    subjs, predicates, objs, scores = objs_scene
    objects_sentences = []
    for person in faces_scene:
        triplets = zip(subjs, predicates, objs)

        # Find triplets for this person
        triplets_person = [(sub, pred, obj) for sub, pred, obj in triplets if (sub == person or obj == person)]
        if len(triplets_person) < 1:
            continue
        this_subj, this_predicate, this_obj = zip(*triplets_person)

        # Group triplets with same predicate together
        unique_pred = set(this_predicate)
        for p in unique_pred:

            # Find all triplets of a character
            this_p_subj, this_p_predicate, this_p_obj = zip(*[(sub, pred, obj) for sub, pred, obj in
                  zip(this_subj, this_predicate, this_obj) if pred == p])

            # Write one sentence per predicate
            this_p_subj = list(set(this_p_subj))
            this_p_obj = list(set(this_p_obj))

            if len(this_p_subj) > 1:
                sentence_subject = ', '.join(this_p_subj[0:-1]) + ' and ' + this_p_subj[-1]
            else:
                sentence_subject = this_p_subj[0]

            if len(this_p_obj) > 1:
                sentence_object = ', '.join(this_p_obj[0:-1]) + ' and ' + this_p_obj[-1]
            else:
                sentence_object = this_p_obj[0]

            sentence = '%s %s %s.' % (sentence_subject.capitalize(), p, sentence_object)
            objects_sentences.append(sentence)
    objects_sentences = ' '.join(objects_sentences)

    # Final description
    vis_description = faces_sentence + actions_sentence + places_sentence + objects_sentences
    return vis_description



def create_description_onlygraph(faces_scene, place_scene, action_scene, objs_scene):

    # Characters
    faces_sentence = ' '.join(faces_scene)

    # Places
    places_sentence =  place_scene


    # Actions
    actions_sentence = action_scene[0].lower() + action_scene[1:]

    # Objects sentences
    subjs, predicates, objs, scores = objs_scene
    objects_sentences = []
    for person in faces_scene:
        triplets = zip(subjs, predicates, objs)

        # Find triplets for this person
        triplets_person = [(sub, pred, obj) for sub, pred, obj in triplets if (sub == person or obj == person)]
        if len(triplets_person) < 1:
            continue
        this_subj, this_predicate, this_obj = zip(*triplets_person)

        for s, p, o in zip(this_subj, this_predicate, this_obj):
            sentence = '%s %s %s.' % (s, p, o)
            objects_sentences.append(sentence)

    objects_sentences = ' '.join(objects_sentences)

    # Final description
    vis_description = faces_sentence + ' ' + actions_sentence + ' ' + places_sentence + ' ' + objects_sentences
    return vis_description


def generate_scene_description(args):

    # Prepare data
    allscenes = get_all_scenes_dataset(args.dataset)
    df_characters, df_places, action_scene_order, action_preds, actions_vocab, objs_dict, preds_dict = load_visual_features(args)

    with open(os.path.join(args.data_dir, args.output_file), 'w') as fout:
        fout.write('Scene\tDescription\n')

        for scenename in tqdm(allscenes, desc="Scenes"):

            # Get all the frames in the scene
            [episode, _, sframe, eframe] = scenename.split('_')
            frameslist = [os.path.join(episode, 'frame_%04d.jpeg' % framenum) for framenum in list(range(int(sframe), int(eframe)+1))]

            # Visual info of this scene
            this_list_characters = get_characters_scene(df_characters, frameslist)
            this_place = get_place_scene(df_places, frameslist)
            this_objs = get_objrelations_scene(episode, frameslist, df_characters, objs_dict, preds_dict)
            this_action = get_mainaction_scene(action_scene_order, scenename, action_preds, actions_vocab)

            # Generate description with visual info
            vis_description = create_description(this_list_characters, this_place, this_action, this_objs)
            fout.write('%s\t%s\n' % (scenename, vis_description))

        fout.close()


if __name__ == "__main__":

    args = get_params()

    # Check dataset
    assert args.dataset in ['knowit', 'tvqa']
    if args.dataset == 'knowit':
        args.data_dir = os.path.join(args.data_dir, 'knowit_observe')
    elif args.dataset == 'tvqa':
        # args.data_dir = os.path.join(args.data_dir, 'tvqa_observe')
        logger.error('Sorry, TVQA+ dataset not implemented yet.')
        import sys
        sys.exit(0)

    generate_scene_description(args)