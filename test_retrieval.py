# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates the retrieval model."""
import numpy as np
import torch
import random
from tqdm import tqdm as tqdm
from collections import OrderedDict


def fiq_test(opt, model, testset):
    model.eval()

    all_imgs = []
    all_queries = []
    all_target_captions = []
    all_target_ids = []

    imgs = []
    mods = []
    out = []

    for i in tqdm(range(len(testset))):
        torch.cuda.empty_cache()
        item = testset[i]
        imgs += [testset.get_img(item['source_img_id'])]

        all_target_captions += [item['target_caption']]
        all_target_ids += [item['target_image_name']]

        mods += [item['target_caption']]

        if len(imgs) >= opt.batch_size or i == len(testset) - 1:
            imgs = torch.stack(imgs).float()
            imgs = torch.autograd.Variable(imgs).cuda()

            dct_with_representations = model.compose_img_text(imgs.cuda(), mods)
            f = dct_with_representations["repres"].data.cpu().numpy()
            all_queries += [f]

            imgs = []
            mods = []

    all_queries = np.concatenate(all_queries)
    print("all_queries len:", len(all_queries))

    # compute all image features
    imgs = []
    for i, original_image_id in enumerate(tqdm(testset.all_imgs_from_cat)):
        imgs += [testset.get_img_from_split(original_image_id)]
        if len(imgs) >= opt.batch_size or i == len(testset.all_imgs_from_cat) - 1:
            imgs = torch.stack(imgs).float()
            imgs = torch.autograd.Variable(imgs).cuda()
            imgs = model.extract_img_feature(imgs.cuda()).data.cpu().numpy()

            all_imgs += [imgs]
            imgs = []

    all_imgs = np.concatenate(all_imgs)

    print("all_imgs len:", len(all_imgs))
    print("all_imgs_from_category len:", len(testset.all_imgs_from_cat))

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    sims = all_queries.dot(all_imgs.T)
    print("sims shape: ", sims.shape)

    nn_result = [np.argsort(-sims[i, :])[:150] for i in range(sims.shape[0])]  # take more to remove duplicates
    nn_result_ids = [[testset.all_imgs_from_cat[nn] for nn in nns] for nns in nn_result]

    filtered_ids = []
    for ranking_ids in nn_result_ids:
        filtered_id_50 = list(OrderedDict.fromkeys(ranking_ids))[:50]  # filter duplicates and preserve order
        filtered_ids.append(filtered_id_50)

    if opt.category_to_train == 'all':
        cats_recalls = {'dress': {'recall': 0.0, 'num': 0},
                        'shirt': {'recall': 0.0, 'num': 0},
                        'toptee': {'recall': 0.0, 'num': 0}}

        things = {}

        for i, target_caption in enumerate(all_target_captions):
            clothing = target_caption.split()[0]
            if clothing in things:
                things[clothing].append({'orig_index': i, 'target_caption': target_caption})
            else:
                things[clothing] = []
                things[clothing].append({'orig_index': i, 'target_caption': target_caption})

        cats_recalls['dress']['num'] = len(things['dress'])
        cats_recalls['shirt']['num'] = len(things['shirt'])
        cats_recalls['toptee']['num'] = len(things['toptee'])
    else:
        cats_recalls = {opt.category_to_train: {'recall': 0.0, 'num': 0}}
        cats_recalls[opt.category_to_train]['num'] = len(testset)

    for k in [1, 5, 10, 50, 100]:
        for i, nns in enumerate(filtered_ids):
            if all_target_ids[i] in nns[:k]:
                if opt.category_to_train == 'all':
                    cats_recalls[all_target_captions[i].split()[0]]['recall'] += 1
                else:
                    cats_recalls[opt.category_to_train]['recall'] += 1
        for cat in cats_recalls:
            cats_recalls[cat]['recall'] /= cats_recalls[cat]['num']
            out += [('recall_top' + str(k) + '_correct_' + cat, cats_recalls[cat]['recall'])]

    return out


def test(opt, model, testset):
    """Tests a model over the given testset."""
    model.eval()
    test_queries = testset.get_test_queries()

    all_imgs = []
    all_captions = []
    all_queries = []
    all_target_captions = []
    if test_queries:
        imgs = []
        mods = []
        for t in tqdm(test_queries):
            torch.cuda.empty_cache()
            imgs += [testset.get_img(t['source_img_id'])]
            if opt.use_complete_text_query:
                if opt.dataset == 'mitstates':
                    mods += [t['mod']['str'] + " " + t["noun"]]
                else:
                    mods += [t['target_caption']]
            else:
                mods += [t['mod']['str']]

            if len(imgs) >= opt.batch_size or t is test_queries[-1]:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()
                dct_with_representations = model.compose_img_text(imgs.cuda(), mods)
                f = dct_with_representations["repres"].data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
        all_queries = np.concatenate(all_queries)
        all_target_captions = [t['target_caption'] for t in test_queries]

        # compute all image features
        imgs = []
        for i in tqdm(range(len(testset.imgs))):
            imgs += [testset.get_img(i)]
            if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs).cuda()
                imgs = model.extract_img_feature(imgs.cuda()).data.cpu().numpy()

                all_imgs += [imgs]
                imgs = []
        all_imgs = np.concatenate(all_imgs)
        all_captions = [img['captions'][0] for img in testset.imgs]

    else:
        # use training queries to approximate training retrieval performance
        imgs0 = []
        imgs = []
        mods = []
        training_approx = 9600
        for i in range(training_approx):
            torch.cuda.empty_cache()
            item = testset[i]
            imgs += [item['source_img_data']]
            if opt.use_complete_text_query:
                if opt.dataset == 'mitstates':
                    mods += [item['mod']['str'] + " " + item["noun"]]
                else:
                    mods += [item['target_caption']]
            else:
                mods += [item['mod']['str']]

            if len(imgs) >= opt.batch_size or i == training_approx:
                imgs = torch.stack(imgs).float()
                imgs = torch.autograd.Variable(imgs)
                dct_with_representations = model.compose_img_text(imgs.cuda(), mods)
                f = dct_with_representations["repres"].data.cpu().numpy()
                all_queries += [f]
                imgs = []
                mods = []
            imgs0 += [item['target_img_data']]
            if len(imgs0) >= opt.batch_size or i == training_approx:
                imgs0 = torch.stack(imgs0).float()
                imgs0 = torch.autograd.Variable(imgs0)
                imgs0 = model.extract_img_feature(imgs0.cuda()).data.cpu().numpy()
                all_imgs += [imgs0]
                imgs0 = []
            all_captions += [item['target_caption']]
            all_target_captions += [item['target_caption']]
        all_imgs = np.concatenate(all_imgs)
        all_queries = np.concatenate(all_queries)

    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    sims = all_queries.dot(all_imgs.T)
    if test_queries:
        for i, t in enumerate(test_queries):
            sims[i, t['source_img_id']] = -10e10  # remove query image
    nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]

    # compute recalls
    out = []
    nn_result = [[all_captions[nn] for nn in nns] for nns in nn_result]
    for k in [1, 5, 10, 50, 100]:
        r = 0.0
        for i, nns in enumerate(nn_result):
            if all_target_captions[i] in nns[:k]:
                r += 1
        r /= len(nn_result)
        out += [('recall_top' + str(k) + '_correct_composition', r)]

        if opt.dataset == 'mitstates':
            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i].split()[0] in [c.split()[0] for c in nns[:k]]:
                    r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_adj', r)]

            r = 0.0
            for i, nns in enumerate(nn_result):
                if all_target_captions[i].split()[1] in [c.split()[1] for c in nns[:k]]:
                    r += 1
            r /= len(nn_result)
            out += [('recall_top' + str(k) + '_correct_noun', r)]

    return out
