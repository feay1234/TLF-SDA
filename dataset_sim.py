# coding=utf-8
#
# Copyright 2020 Heinrich Heine University Duesseldorf
#
# Part of this code is based on the source code of BERT-DST
# (arXiv:1907.03040)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataset_multiwoz21 import normalize_text
from utils_dst import (DSTExample, convert_to_unicode)
import json, random, collections
import re
from random import sample

def dialogue_state_to_sv_dict(sv_list):
    sv_dict = {}
    for d in sv_list:
        sv_dict[d['slot']] = d['value']
    return sv_dict


def get_token_and_slot_label(turn):
    if 'system_utterance' in turn:
        sys_utt_tok = turn['system_utterance']['tokens']
        sys_slot_label = turn['system_utterance']['slots']
    else:
        sys_utt_tok = []
        sys_slot_label = []

    usr_utt_tok = turn['user_utterance']['tokens']
    usr_slot_label = turn['user_utterance']['slots']
    return sys_utt_tok, sys_slot_label, usr_utt_tok, usr_slot_label


def get_tok_label(prev_ds_dict, cur_ds_dict, slot_type, sys_utt_tok,
                  sys_slot_label, usr_utt_tok, usr_slot_label, dial_id,
                  turn_id, slot_last_occurrence=True):
    """The position of the last occurrence of the slot value will be used."""
    sys_utt_tok_label = [0 for _ in sys_utt_tok]
    usr_utt_tok_label = [0 for _ in usr_utt_tok]
    if slot_type not in cur_ds_dict:
        class_type = 'none'
    else:
        value = cur_ds_dict[slot_type]
        if value == 'dontcare' and (slot_type not in prev_ds_dict or prev_ds_dict[slot_type] != 'dontcare'):
            # Only label dontcare at its first occurrence in the dialog
            class_type = 'dontcare'
        else: # If not none or dontcare, we have to identify whether
            # there is a span, or if the value is informed
            in_usr = False
            in_sys = False
            for label_d in usr_slot_label:
                if label_d['slot'] == slot_type and value == ' '.join(
                        usr_utt_tok[label_d['start']:label_d['exclusive_end']]):

                    for idx in range(label_d['start'], label_d['exclusive_end']):
                        usr_utt_tok_label[idx] = 1
                    in_usr = True
                    class_type = 'copy_value'
                    if slot_last_occurrence:
                        break
            if not in_usr or not slot_last_occurrence:
                for label_d in sys_slot_label:
                    if label_d['slot'] == slot_type and value == ' '.join(
                            sys_utt_tok[label_d['start']:label_d['exclusive_end']]):
                        for idx in range(label_d['start'], label_d['exclusive_end']):
                            sys_utt_tok_label[idx] = 1
                        in_sys = True
                        class_type = 'inform'
                        if slot_last_occurrence:
                            break
            if not in_usr and not in_sys:
                assert sum(usr_utt_tok_label + sys_utt_tok_label) == 0
                if (slot_type not in prev_ds_dict or value != prev_ds_dict[slot_type]):
                    raise ValueError('Copy value cannot found in Dial %s Turn %s' % (str(dial_id), str(turn_id)))
                else:
                    class_type = 'none'
            else:
                assert sum(usr_utt_tok_label + sys_utt_tok_label) > 0
    return sys_utt_tok_label, usr_utt_tok_label, class_type


def delex_utt(utt, values):
    utt_delex = utt.copy()
    for v in values:
        utt_delex[v['start']:v['exclusive_end']] = ['[UNK]'] * (v['exclusive_end'] - v['start'])
    return utt_delex
    

def get_turn_label(turn, prev_dialogue_state, slot_list, dial_id, turn_id,
                   delexicalize_sys_utts=False, slot_last_occurrence=True):
    """Make turn_label a dictionary of slot with value positions or being dontcare / none:
    Turn label contains:
      (1) the updates from previous to current dialogue state,
      (2) values in current dialogue state explicitly mentioned in system or user utterance."""
    prev_ds_dict = dialogue_state_to_sv_dict(prev_dialogue_state)
    cur_ds_dict = dialogue_state_to_sv_dict(turn['dialogue_state'])

    (sys_utt_tok, sys_slot_label, usr_utt_tok, usr_slot_label) = get_token_and_slot_label(turn)

    sys_utt_tok_label_dict = {}
    usr_utt_tok_label_dict = {}
    inform_label_dict = {}
    inform_slot_label_dict = {}
    referral_label_dict = {}
    class_type_dict = {}

    for slot_type in slot_list:
        inform_label_dict[slot_type] = 'none'
        inform_slot_label_dict[slot_type] = 0
        referral_label_dict[slot_type] = 'none' # Referral is not present in sim data
        sys_utt_tok_label, usr_utt_tok_label, class_type = get_tok_label(
            prev_ds_dict, cur_ds_dict, slot_type, sys_utt_tok, sys_slot_label,
            usr_utt_tok, usr_slot_label, dial_id, turn_id,
            slot_last_occurrence=slot_last_occurrence)
        if sum(sys_utt_tok_label) > 0:
            inform_label_dict[slot_type] = cur_ds_dict[slot_type]
            inform_slot_label_dict[slot_type] = 1
        sys_utt_tok_label = [0 for _ in sys_utt_tok_label] # Don't use token labels for sys utt
        sys_utt_tok_label_dict[slot_type] = sys_utt_tok_label
        usr_utt_tok_label_dict[slot_type] = usr_utt_tok_label
        class_type_dict[slot_type] = class_type

    if delexicalize_sys_utts:
        sys_utt_tok = delex_utt(sys_utt_tok, sys_slot_label)

    return (sys_utt_tok, sys_utt_tok_label_dict,
            usr_utt_tok, usr_utt_tok_label_dict,
            inform_label_dict, inform_slot_label_dict,
            referral_label_dict, cur_ds_dict, class_type_dict)

def tokenize(utt):
    utt_lower = convert_to_unicode(utt).lower()
    utt_lower = normalize_text(utt_lower)
    utt_tok = [tok for tok in map(str.strip, re.split("(\W+)", utt_lower)) if len(tok) > 0]
    return utt_tok

def create_examples(input_file, set_type, slot_list,
                    label_maps={},
                    append_history=False,
                    use_history_labels=False,
                    swap_utterances=False,
                    label_value_repetitions=False,
                    delexicalize_sys_utts=False,
                    analyze=False,
                    local_machine=False,
                    seq_num=0,
                    perturbation=1,
                    index=-1,
                    batch_size=48):
    """Read a DST json file into a list of DSTExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)



    examples = []
    for entry in input_data:
        dial_id = entry['dialogue_id']
        domains = dial_id.split("_")[0]
        prev_ds = []
        hst = []
        prev_hst_lbl_dict = {slot: [] for slot in slot_list}
        prev_ds_lbl_dict = {slot: 'none' for slot in slot_list}

        if local_machine:
            if len(examples) == 10:
                break
        memo = {}
        for turn_id, turn in enumerate(entry['turns']):
            guid = '%s-%s-%s' % (set_type, dial_id, str(turn_id))
            did = str(dial_id)

            ds_lbl_dict = prev_ds_lbl_dict.copy()
            hst_lbl_dict = prev_hst_lbl_dict.copy()
            (text_a,
             text_a_label,
             text_b,
             text_b_label,
             inform_label,
             inform_slot_label,
             referral_label,
             cur_ds_dict,
             class_label) = get_turn_label(turn,
                                           prev_ds,
                                           slot_list,
                                           dial_id,
                                           turn_id,
                                           delexicalize_sys_utts=delexicalize_sys_utts,
                                           slot_last_occurrence=True)

            if swap_utterances:
                txt_a = text_b
                txt_b = text_a
                txt_a_lbl = text_b_label
                txt_b_lbl = text_a_label
            else:
                txt_a = text_a
                txt_b = text_b
                txt_a_lbl = text_a_label
                txt_b_lbl = text_b_label

            value_dict = {}

            for slot in slot_list:
                if slot in cur_ds_dict:
                    value_dict[slot] = cur_ds_dict[slot]
                else:
                    value_dict[slot] = 'none'
                if class_label[slot] != 'none':
                    ds_lbl_dict[slot] = class_label[slot]
                if append_history:
                    if use_history_labels:
                        hst_lbl_dict[slot] = txt_a_lbl[slot] + txt_b_lbl[slot] + hst_lbl_dict[slot]
                    else:
                        hst_lbl_dict[slot] = [0 for _ in txt_a_lbl[slot] + txt_b_lbl[slot] + hst_lbl_dict[slot]]

            if set_type != "train" and perturbation > 0:
                _hst = " ".join(hst).split(".")
                # shuffle utterance
                if perturbation == 1:
                    random.shuffle(_hst)
                # reverse utterance
                elif perturbation == 2:
                    _hst = _hst[::-1]
                # shuffle word
                elif perturbation == 3:
                    for i in range(len(_hst)):
                        _hst[i] = " ".join(sample(_hst[i].split(), len(_hst[i].split())))
                # reverse word
                elif perturbation == 4:
                    for i in range(len(_hst)):
                        _hst[i] = " ".join(_hst[i].split()[::-1])


                hst = []
                for p in _hst:
                    hst.extend(p.split())

            examples.append(DSTExample(
                guid=guid,
                did=did,
                text_a=txt_a,
                text_b=txt_b,
                history=hst,
                text_a_label=txt_a_lbl,
                text_b_label=txt_b_lbl,
                history_label=prev_hst_lbl_dict,
                values=value_dict,
                inform_label=inform_label,
                inform_slot_label=inform_slot_label,
                refer_label=referral_label,
                diag_state=prev_ds_lbl_dict,
                class_label=class_label))

            memo[guid] = examples[-1]

            prev_ds = turn['dialogue_state']
            prev_ds_lbl_dict = ds_lbl_dict.copy()
            prev_hst_lbl_dict = hst_lbl_dict.copy()

            if append_history:
                hst = txt_a + txt_b + hst

        # Sequential Data Augmentation (SDA)
        if set_type == "train":
            ids = list(memo.keys())
            for _seq_num in range(1, seq_num + 1):
                for i in range(len(ids) - _seq_num):
                    guid, did = memo[ids[i]].guid, memo[ids[i]].did
                    history = memo[ids[i]].history
                    text_a, text_b = [], []
                    text_a_label, text_b_label = {}, {}
                    inform_label = {slot: 'none' for slot in slot_list}
                    inform_slot_label = {slot: 0 for slot in slot_list}
                    refer_label = {slot: 'none' for slot in slot_list}
                    class_label = {slot: 'none' for slot in slot_list}
                    refer_turn_label = {slot: 0 for slot in slot_list}

                    for j in range(i, i + _seq_num + 1):
                        text_a.extend(memo[ids[j]].text_a)
                        text_b.extend(memo[ids[j]].text_b)

                        for slot in slot_list:
                            if slot not in text_a_label:
                                text_a_label[slot] = memo[ids[j]].text_a_label[slot]
                                text_b_label[slot] = memo[ids[j]].text_b_label[slot]
                            else:
                                text_a_label[slot].extend(memo[ids[j]].text_a_label[slot])
                                text_b_label[slot].extend(memo[ids[j]].text_b_label[slot])

                            if memo[ids[j]].inform_label[slot] != 'none':
                                inform_label[slot] = memo[ids[j]].inform_label[slot]

                            if memo[ids[j]].inform_slot_label[slot] != 0:
                                inform_slot_label[slot] = memo[ids[j]].inform_slot_label[slot]

                            if memo[ids[j]].refer_label[slot] != 'none':
                                refer_label[slot] = memo[ids[j]].refer_label[slot]

                            if memo[ids[j]].class_label[slot] != 'none':
                                class_label[slot] = memo[ids[j]].class_label[slot]

                    history_label = memo[ids[j]].history_label.copy()
                    values = memo[ids[j]].values.copy()
                    diag_state = memo[ids[j]].diag_state.copy()

                    example = DSTExample(
                        guid=guid,
                        did=did,
                        text_a=text_a,
                        text_b=text_b,
                        history=history,
                        text_a_label=text_a_label,
                        text_b_label=text_b_label,
                        history_label=history_label,
                        values=values,
                        inform_label=inform_label,
                        inform_slot_label=inform_slot_label,
                        refer_label=refer_label,
                        diag_state=diag_state,
                        class_label=class_label,
                        refer_turn_label=refer_turn_label)

                    examples.append(example)


    return examples
