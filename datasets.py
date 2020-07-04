# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Great things."""

"""Provides data for training and testing."""
import numpy as np
import PIL
import skimage
import torch
import json
import torch.utils.data
import torchvision
import warnings
import random


class BaseDataset(torch.utils.data.Dataset):
    """Base class for a dataset."""

    def __init__(self):
        super(BaseDataset, self).__init__()
        self.imgs = []
        self.test_queries = []

    def get_loader(self,
                   batch_size,
                   shuffle=False,
                   drop_last=False,
                   num_workers=0):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=lambda i: i)

    def get_test_queries(self):
        return self.test_queries

    def get_all_texts(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        return self.generate_random_query_target()

    def generate_random_query_target(self):
        raise NotImplementedError

    def get_img(self, idx, raw_img=False):
        raise NotImplementedError

class Fashion200k(BaseDataset):
    """Fashion200k dataset."""

    def __init__(self, path, split='train', transform=None):
        super(Fashion200k, self).__init__()

        self.split = split
        self.transform = transform
        self.img_path = path + '/'

        # get label files for the split
        label_path = path + '/labels/'
        from os import listdir
        from os.path import isfile
        from os.path import join
        label_files = [
            f for f in listdir(label_path) if isfile(join(label_path, f))
        ]
        label_files = [f for f in label_files if split in f]

        # read image info from label files
        self.imgs = []

        def caption_post_process(s):
            return s.strip().replace('.',
                                     'dotmark').replace('?', 'questionmark').replace(
                '&', 'andmark').replace('*', 'starmark')

        for filename in label_files:
            print('read ' + filename)
            with open(label_path + '/' + filename) as f:
                lines = f.readlines()
            for line in lines:
                line = line.split('	')
                img = {
                    'file_path': line[0],
                    'detection_score': line[1],
                    'captions': [caption_post_process(line[2])],
                    'split': split,
                    'modifiable': False
                }
                self.imgs += [img]
        print('Fashion200k:', len(self.imgs), 'images')

        # generate query for training or testing
        if split == 'train':
            self.caption_index_init_()
        else:
            self.generate_test_queries_()

    def get_different_word(self, source_caption, target_caption):
        source_words = source_caption.split()
        target_words = target_caption.split()
        for source_word in source_words:
            if source_word not in target_words:
                break
        for target_word in target_words:
            if target_word not in source_words:
                break
        mod_str = 'replace ' + source_word + ' with ' + target_word
        return source_word, target_word, mod_str

    def generate_test_queries_(self):
        file2imgid = {}
        for i, img in enumerate(self.imgs):
            file2imgid[img['file_path']] = i
        with open(self.img_path + '/test_queries.txt') as f:
            lines = f.readlines()
        self.test_queries = []
        for line in lines:
            source_file, target_file = line.split()
            idx = file2imgid[source_file]
            target_idx = file2imgid[target_file]
            source_caption = self.imgs[idx]['captions'][0]
            target_caption = self.imgs[target_idx]['captions'][0]
            source_word, target_word, mod_str = self.get_different_word(
                source_caption, target_caption)
            self.test_queries += [{
                'source_img_id': idx,
                'source_caption': source_caption,
                'target_caption': target_caption,
                'mod': {
                    'str': mod_str
                }
            }]

    def caption_index_init_(self):
        """ index caption to generate training query-target example on the fly later"""

        # index caption 2 caption_id and caption 2 image_ids
        caption2id = {}
        id2caption = {}
        caption2imgids = {}
        for i, img in enumerate(self.imgs):
            for c in img['captions']:
                if c not in caption2id:
                    id2caption[len(caption2id)] = c
                    caption2id[c] = len(caption2id)
                    caption2imgids[c] = []
                caption2imgids[c].append(i)
        self.caption2imgids = caption2imgids
        print(len(caption2imgids), 'unique cations')

        # parent captions are 1-word shorter than their children
        parent2children_captions = {}
        for c in caption2id.keys():
            for w in c.split():
                p = c.replace(w, '')
                p = p.replace('  ', ' ').strip()
                if p not in parent2children_captions:
                    parent2children_captions[p] = []
                if c not in parent2children_captions[p]:
                    parent2children_captions[p].append(c)
        self.parent2children_captions = parent2children_captions

        # identify parent captions for each image
        for img in self.imgs:
            img['modifiable'] = False
            img['parent_captions'] = []
        for p in parent2children_captions:
            if len(parent2children_captions[p]) >= 2:
                for c in parent2children_captions[p]:
                    for imgid in caption2imgids[c]:
                        self.imgs[imgid]['modifiable'] = True
                        self.imgs[imgid]['parent_captions'] += [p]
        num_modifiable_imgs = 0
        for img in self.imgs:
            if img['modifiable']:
                num_modifiable_imgs += 1
        print('Modifiable images', num_modifiable_imgs)

    def caption_index_sample_(self, idx):
        while not self.imgs[idx]['modifiable']:
            idx = np.random.randint(0, len(self.imgs))

        # find random target image (same parent)
        img = self.imgs[idx]
        while True:
            p = random.choice(img['parent_captions'])
            c = random.choice(self.parent2children_captions[p])
            if c not in img['captions']:
                break
        target_idx = random.choice(self.caption2imgids[c])

        # find the word difference between query and target (not in parent caption)
        source_caption = self.imgs[idx]['captions'][0]
        target_caption = self.imgs[target_idx]['captions'][0]
        source_word, target_word, mod_str = self.get_different_word(
            source_caption, target_caption)
        return idx, target_idx, source_word, target_word, mod_str

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            for c in img['captions']:
                texts.append(c)
        return texts

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        idx, target_idx, source_word, target_word, mod_str = self.caption_index_sample_(
            idx)
        out = {}
        out['source_img_id'] = idx
        out['source_img_data'] = self.get_img(idx)
        out['source_caption'] = self.imgs[idx]['captions'][0]
        out['target_img_id'] = target_idx
        out['target_img_data'] = self.get_img(target_idx)
        out['target_caption'] = self.imgs[target_idx]['captions'][0]
        out['mod'] = {'str': mod_str}
        return out

    def get_img(self, idx, raw_img=False):
        img_path = self.img_path + self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img

class MITStates(BaseDataset):
    """MITStates dataset."""

    def __init__(self, path, split='train', transform=None):
        super(MITStates, self).__init__()
        self.path = path
        self.transform = transform
        self.split = split

        self.imgs = []
        test_nouns = [
            u'armor', u'bracelet', u'bush', u'camera', u'candy', u'castle',
            u'ceramic', u'cheese', u'clock', u'clothes', u'coffee', u'fan', u'fig',
            u'fish', u'foam', u'forest', u'fruit', u'furniture', u'garden', u'gate',
            u'glass', u'horse', u'island', u'laptop', u'lead', u'lightning',
            u'mirror', u'orange', u'paint', u'persimmon', u'plastic', u'plate',
            u'potato', u'road', u'rubber', u'sand', u'shell', u'sky', u'smoke',
            u'steel', u'stream', u'table', u'tea', u'tomato', u'vacuum', u'wax',
            u'wheel', u'window', u'wool'
        ]

        from os import listdir
        for f in listdir(path + '/images'):
            if ' ' not in f:
                continue
            adj, noun = f.split()
            if adj == 'adj':
                continue
            if split == 'train' and noun in test_nouns:
                continue
            if split == 'test' and noun not in test_nouns:
                continue

            for file_path in listdir(path + '/images/' + f):
                assert (file_path.endswith('jpg'))
                self.imgs += [{
                    'file_path': path + '/images/' + f + '/' + file_path,
                    'captions': [f],
                    'adj': adj,
                    'noun': noun
                }]

        self.caption_index_init_()
        if split == 'test':
            self.generate_test_queries_()

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            texts += img['captions']
        return texts

    def __getitem__(self, idx):
        try:
            self.saved_item
        except:
            self.saved_item = None
        if self.saved_item is None:
            while True:
                idx, target_idx1 = self.caption_index_sample_(idx)
                idx, target_idx2 = self.caption_index_sample_(idx)
                if self.imgs[target_idx1]['adj'] != self.imgs[target_idx2]['adj']:
                    break
            idx, target_idx = [idx, target_idx1]
            self.saved_item = [idx, target_idx2]
        else:
            idx, target_idx = self.saved_item
            self.saved_item = None

        mod_str = self.imgs[target_idx]['adj']

        return {
            'source_img_id': idx,
            'source_img_data': self.get_img(idx),
            'source_caption': self.imgs[idx]['captions'][0],
            'target_img_id': target_idx,
            'target_img_data': self.get_img(target_idx),
            'noun': self.imgs[idx]['noun'],
            'target_caption': self.imgs[target_idx]['captions'][0],
            'mod': {
                'str': mod_str
            }
        }

    def caption_index_init_(self):
        self.caption2imgids = {}
        self.noun2adjs = {}
        for i, img in enumerate(self.imgs):
            cap = img['captions'][0]
            adj = img['adj']
            noun = img['noun']
            if cap not in self.caption2imgids.keys():
                self.caption2imgids[cap] = []
            if noun not in self.noun2adjs.keys():
                self.noun2adjs[noun] = []
            self.caption2imgids[cap].append(i)
            if adj not in self.noun2adjs[noun]:
                self.noun2adjs[noun].append(adj)
        for noun, adjs in self.noun2adjs.items():
            assert len(adjs) >= 2

    def caption_index_sample_(self, idx):
        noun = self.imgs[idx]['noun']
        # adj = self.imgs[idx]['adj']
        target_adj = random.choice(self.noun2adjs[noun])
        target_caption = target_adj + ' ' + noun
        target_idx = random.choice(self.caption2imgids[target_caption])
        return idx, target_idx

    def generate_test_queries_(self):
        self.test_queries = []
        for idx, img in enumerate(self.imgs):
            adj = img['adj']
            noun = img['noun']
            for target_adj in self.noun2adjs[noun]:
                if target_adj != adj:
                    mod_str = target_adj
                    self.test_queries += [{
                        'source_img_id': idx,
                        'source_caption': adj + ' ' + noun,
                        'target_caption': target_adj + ' ' + noun,
                        'noun': self.imgs[idx]['noun'],
                        'mod': {
                            'str': mod_str
                        }
                    }]
        print(len(self.test_queries), 'test queries')

    def __len__(self):
        return len(self.imgs)

    def get_img(self, idx, raw_img=False):
        img_path = self.imgs[idx]['file_path']
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')
        if raw_img:
            return img
        if self.transform:
            img = self.transform(img)
        return img

class FashionIQ(BaseDataset):
    """FashionIQ dataset."""

    def __init__(self, path, split='train', cat_type='all', transform=None):
        super(FashionIQ, self).__init__()
        self.path = path
        self.transform = transform
        self.split = split
        self.imgs = []

        caps_path = path + "captions/"

        train_caps = ["cap.dress.train.json",
                      "cap.shirt.train.json",
                      "cap.toptee.train.json"]

        val_caps = ["cap.dress.val.json",
                    "cap.shirt.val.json",
                    "cap.toptee.val.json"]

        # load all pool of images
        if cat_type == 'all':
            self.all_imgs_from_cat = []
            for c in ['dress', 'shirt', 'toptee']:
                with open(path + 'image_splits/split.' + c + '.' + split + '.json') as f:
                    self.all_imgs_from_cat += json.load(f)

        #  load splits
        caps = []
        if split == 'val':
            print("Using " + cat_type + " val data")
            if cat_type == 'all':
                caps = val_caps
            else:
                caps = ["cap." + cat_type + ".val.json"]
                with open(path + 'image_splits/split.' + cat_type + '.val.json') as f:
                    self.all_imgs_from_cat = json.load(f)
        elif split == 'train':
            print("Using " + cat_type + " train data")
            if cat_type == 'all':
                caps = train_caps
            else:
                caps = ["cap." + cat_type + ".train.json"]
                with open(path + 'image_splits/split.' + cat_type + '.train.json') as f:
                    self.all_imgs_from_cat = json.load(f)

        for cat in caps:
            with open(caps_path + cat) as f:
                cap2smth = json.load(f)
            cat_name = cat.split('.')[1]
            for idx, cap in enumerate(cap2smth):
                if cat_type == 'all':
                    captions = (cat_name + ' ' + ' and it '.join(cap['captions'])).lower()
                else:
                    captions = ' '.join(cap['captions']).lower()
                d = {
                    'source_image_path': path + 'all_imgs/' + cap['candidate'] + '.jpg',
                    'captions': captions,
                    'original_captions': cap['captions'],
                    'candidate_image_name': cap['candidate'],
                    'source_img_id': idx,
                }
                if split != 'real_test':
                    d['target_image_path'] = path + 'all_imgs/' + cap['target'] + '.jpg'
                    d['target_image_name'] = cap['target']

                self.imgs += [d]

    def get_all_texts(self):
        texts = []
        for img in self.imgs:
            texts += img['captions']
        return texts

    def __getitem__(self, idx):
        d = {
            'source_img_id': idx,
            'source_img_data': self.get_img(idx, if_target=False),
            'target_caption': self.imgs[idx]['captions'],
            'original_captions': self.imgs[idx]['original_captions'],
            'candidate_image_name': self.imgs[idx]['candidate_image_name'],
            'mod': {
                'str': self.imgs[idx]['captions']
            }
        }

        if self.split != 'real_test':
            d['target_image_name'] = self.imgs[idx]['target_image_name']
            d['target_img_data'] = self.get_img(idx, if_target=True)

        return d

    def __len__(self):
        return len(self.imgs)

    def get_img(self, idx, if_target=False):
        if if_target:
            img_path = self.imgs[idx]['target_image_path']
        else:
            img_path = self.imgs[idx]['source_image_path']

        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img

    def get_img_from_split(self, original_image_id):
        img_path = self.path + 'all_imgs/' + original_image_id + '.jpg'

        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img
