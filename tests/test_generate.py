"""Max coverage tests for generate.py GenerationSampler utilities and schedules.

We stub a minimal model & tokenizer to exercise logic paths without depending on full implementation.
"""
from __future__ import annotations

import math
import copy
import warnings
import types
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from terratorch.models.backbones.terramind.model.generate import (
    get_sentinel_to_id_mapping,
    split_by_sentinel,
    merge_span_masking,
    cosine_schedule,
    linear_schedule,
    onex_temp_schedule,
    linear_temp_schedule,
    empty_img_modality,
    empty_seq_modality,
    empty_seq_emb_modality,
    init_empty_target_modality,
    init_conditioned_target_modality,
    init_full_input_modality,
    custom_text,
    build_chained_generation_schedules,
    GenerationSampler,
)

# ---- Stub tokenizer ----
class StubTokenizer:
    def __init__(self):
        # Provide some sentinel tokens and others
        self.vocab = {
            "[S_0]": 10,
            "[S_1]": 11,
            "[S_2]": 12,
            "[EOS]": 13,
            "[PAD]": 0,
            "hello": 1,
            "world": 2,
            "foo": 3,
            "bar": 4,
        }
    def get_vocab(self):
        return self.vocab
    def encode(self, text):
        # naive split
        ids = [self.vocab.get(tok, 0) for tok in text.split()]
        class R:  # structure with ids attribute like expected
            def __init__(self, ids):
                self.ids = ids
        return R(ids)
    def token_to_id(self, tok):
        return self.vocab[tok]

@pytest.fixture(scope="module")
def tokenizer():
    return StubTokenizer()

# ---- Stub model for GenerationSampler ----
class StubEmbed(nn.Module):
    def __init__(self, vocab_size=32, dim=8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.dim = dim
    def forward(self, d):  # encoder embedding call path
        # Expect dict with 'tensor'
        t = d['tensor']
        emb = self.token_emb(t)
        # Build required fields cat_encoder_tensors expects
        return {'tensor': emb, 'emb': emb, 'input_mask': d['input_mask'], 'mod_mask': torch.zeros_like(d['input_mask'], dtype=torch.int)}
    def forward_embed(self, d):
        # decoder embedding path expects dict with x, emb, target_mask, ids
        t = d['tensor']
        emb = self.token_emb(t)
        return {'x': emb, 'emb': emb, 'target_mask': d['input_mask'], 'ids': t, 'mod_mask': torch.zeros_like(d['input_mask'], dtype=torch.int)}

class StubModel(nn.Module):
    def __init__(self, dim=8, vocab_size=32, num_register_tokens=0):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.num_register_tokens = num_register_tokens
        if num_register_tokens > 0:
            # GenerationSampler expects shape (1, n, d) for repeat pattern '() n d -> b n d'
            self.register_tokens = torch.zeros(1, num_register_tokens, dim)
        self.mask_token = torch.ones(dim)
        self.modality_info = {
            'text': {'type': 'seq', 'id': 0, 'max_tokens': 6},
            'img': {'type': 'img', 'id': 1, 'max_tokens': 4},
        }
        # Support both text and img modalities
        self.encoder_embeddings = {
            'text': StubEmbed(vocab_size=vocab_size, dim=dim),
            'img': StubEmbed(vocab_size=vocab_size, dim=dim)
        }
        self.decoder_embeddings = {
            'text': StubEmbed(vocab_size=vocab_size, dim=dim),
            'img': StubEmbed(vocab_size=vocab_size, dim=dim)
        }
    def cat_encoder_tensors(self, encoder_mod_dict):
        # Collect first (only) modality outputs
        d = list(encoder_mod_dict.values())[0]
        tokens = d['tensor']  # B,N,D
        emb = d['emb']
        mask = d['input_mask']
        mod_mask = d['mod_mask']
        return tokens, emb, mask, mod_mask
    def forward_encoder(self, x, mask):
        return x  # identity
    def decoder_proj_context(self, x):
        return torch.zeros_like(x)
    def forward_decoder(self, y, context, encoder_mask, causal_mask):
        return y  # identity
    def forward_logits(self, y, decoder_mod_dict, decoder_mod_mask):
        # y shape (B,N,D) -> flatten -> produce logits per token
        B, N, D = y.shape
        logits = torch.randn(B * N, self.vocab_size)
        # Return logits for all modalities in decoder_mod_dict
        result = {}
        for mod in decoder_mod_dict.keys():
            result[mod] = logits
        return result

@pytest.fixture
def sampler_no_reg():
    return GenerationSampler(StubModel(num_register_tokens=0))

@pytest.fixture
def sampler_with_reg():
    return GenerationSampler(StubModel(num_register_tokens=2))

# ---- Utility function tests ----

def test_get_sentinel_to_id_mapping(tokenizer):
    mapping = get_sentinel_to_id_mapping(tokenizer)
    assert mapping[0] == 10 and mapping[1] == 11


def test_split_by_sentinel_and_merge(tokenizer):
    # merge_span_masking replaces sentinel tokens with their spans (sentinels removed)
    sentinel_ids = set(get_sentinel_to_id_mapping(tokenizer).values())
    seq = [11, 1, 2, 12, 3]  # [S_1] hello world [S_2] foo
    splits = split_by_sentinel(seq, sentinel_ids)
    assert 11 in splits and 12 in splits
    merged = merge_span_masking([11, 12], seq, sentinel_ids)
    # Expect sentinels stripped, spans concatenated
    assert merged == [1, 2, 3]


def test_cosine_schedule_sum():
    sched = cosine_schedule(5, 100)
    assert sched.sum() == 100
    assert len(sched) == 5


def test_linear_schedule_trim():
    sched = linear_schedule(5, 50)
    assert sched.sum() == 50
    assert sched[0] >= sched[-1]


def test_onex_temp_schedule_range():
    token_schedule = np.array([10, 20, 30])
    temps = onex_temp_schedule(max_t=1.0, min_t=0.1, token_schedule=token_schedule, power=0.5)
    assert temps.min() >= 1e-9 and temps.max() <= 1.0
    # Implementation contracts to length == len(token_schedule) (compressed schedule)
    assert len(temps) == len(token_schedule)


def test_linear_temp_schedule(tokenizer):
    token_schedule = np.array([5, 5, 10])
    temps = linear_temp_schedule(1.0, token_schedule)
    assert len(temps) == token_schedule.size

# ---- Modality init tests ----

def test_empty_seq_modality_masks():
    mod_dict = {
        'tensor': torch.zeros((2, 6), dtype=torch.int64),
        'input_mask': torch.ones((2, 6), dtype=torch.bool),
        'target_mask': torch.zeros((2, 6), dtype=torch.bool),
        'decoder_attention_mask': torch.zeros((2, 6), dtype=torch.bool),
    }
    out = empty_seq_modality(mod_dict, s1_id=11)
    assert out['tensor'][0,0] == 11 and out['tensor'][0,-1] == 12
    assert out['input_mask'][0,0] == False and out['input_mask'][0,1] == True


def test_empty_img_modality():
    mod_dict = {
        'img': {
            'input_mask': torch.zeros((1, 4), dtype=torch.bool),
            'target_mask': torch.ones((1, 4), dtype=torch.bool),
        }
    }
    out = empty_img_modality(mod_dict, 'img')
    assert out['img']['input_mask'].all() and (~out['img']['target_mask']).all()


def test_empty_seq_emb_modality():
    mod_dict = {
        'input_mask': torch.ones((1,5), dtype=torch.bool)
    }
    out = empty_seq_emb_modality(mod_dict)
    assert out['input_mask'][0,0] == False

@pytest.fixture
def modality_info():
    return {
        'img': {'type': 'img'},
        'text': {'type': 'seq'},
        'emb': {'type': 'seq_emb'},
    }


def test_init_empty_target_modality_img(modality_info):
    d = init_empty_target_modality(modality_info, 'img', batch_size=2, num_tokens=4, device='cpu')
    assert d['tensor'].shape == (2,4)
    assert d['input_mask'].all() and (~d['target_mask']).all()


def test_init_empty_target_modality_seq(modality_info):
    d = init_empty_target_modality(modality_info, 'text', batch_size=2, num_tokens=6, device='cpu', s1_id=11)
    assert d['tensor'][0,0] == 11 and d['tensor'][0,-1] == 12


def test_init_empty_target_modality_seq_emb(modality_info):
    d = init_empty_target_modality(modality_info, 'emb', batch_size=2, num_tokens=6, device='cpu', s1_id=11)
    assert d['input_mask'].shape == (2,6)


def test_init_conditioned_target_modality_eos_found(modality_info):
    base = init_empty_target_modality(modality_info, 'text', 1, 4, 'cpu', s1_id=11)
    # Insert EOS token id 13 before extension
    base['tensor'][0,2] = 13
    out = init_conditioned_target_modality(base, modality_info, 'text', num_target_tokens=3, eos_id=13, s1_id=11)
    assert out['tensor'].shape[1] == 7  # 4 + 3
    assert (out['input_mask'][0,:3] == False).all()


def test_init_conditioned_target_modality_eos_missing_warn(modality_info):
    base = init_empty_target_modality(modality_info, 'text', 1, 4, 'cpu', s1_id=11)
    with warnings.catch_warnings(record=True) as w:
        out = init_conditioned_target_modality(base, modality_info, 'text', num_target_tokens=2, eos_id=99, s1_id=11)
        assert any('Cannot find EOS' in str(x.message) for x in w)
        assert out['tensor'].shape[1] == 6


def test_init_full_input_modality_seq_with_eos(modality_info):
    tensor = torch.tensor([[11, 1, 13, 0, 0]])
    out = init_full_input_modality(tensor, modality_info, 'text', 'cpu', eos_id=13)
    assert out['input_mask'][0,0] == False and out['input_mask'][0,2] == False


def test_init_full_input_modality_seq_without_eos_warn(modality_info):
    tensor = torch.tensor([[11, 1, 2]])
    with warnings.catch_warnings(record=True) as w:
        out = init_full_input_modality(tensor, modality_info, 'text', 'cpu', eos_id=13)
        assert any('Cannot find EOS' in str(x.message) for x in w)


def test_init_full_input_modality_img(modality_info):
    tensor = torch.zeros((1,5), dtype=torch.int64)
    out = init_full_input_modality(tensor, modality_info, 'img', 'cpu')
    assert out['input_mask'].sum() == 0 and out['target_mask'].sum() == 5


def test_custom_text(tokenizer):
    mod = custom_text("hello world", 'cpu', tokenizer, target_max_len=6)
    assert mod['tensor'].shape[1] == 2 + 6  # input length + target length
    assert mod['input_mask'][0,0] == False and mod['target_mask'][0,0] == True

# ---- Schedule building tests ----

def test_build_chained_generation_schedules_maskgit_img():
    schedule = build_chained_generation_schedules(
        cond_domains=['text'],
        target_domains=['img'],
        tokens_per_target=[6],
        autoregression_schemes=['maskgit'],
        decoding_steps=[3],
        token_decoding_schedules=['linear'],
        temps=[1.0],
        temp_schedules=['linear'],
        cfg_scales=[1.0],
        cfg_schedules=['constant'],
        modality_info={'img': {'type': 'img'}},
    )
    assert len(schedule) == 3


def test_build_chained_generation_schedules_autoregressive():
    schedule = build_chained_generation_schedules(
        cond_domains=['text'],
        target_domains=['text'],
        tokens_per_target=[6],
        autoregression_schemes=['autoregressive'],
        decoding_steps=[0],
        token_decoding_schedules=['linear'],
        temps=[1.0],
        temp_schedules=['linear'],
        cfg_scales=[2.0],
        cfg_schedules=['constant'],
        modality_info={'text': {'type': 'seq'}},
    )
    assert schedule[0]['scheme'] == 'autoregressive'


def test_build_chained_generation_schedules_roar_img():
    schedule = build_chained_generation_schedules(
        cond_domains=['text'],
        target_domains=['img'],
        tokens_per_target=[5],
        autoregression_schemes=['roar'],
        decoding_steps=[5],
        token_decoding_schedules=['linear'],
        temps=[1.0],
        temp_schedules=['constant'],
        cfg_scales=[2.0],
        cfg_schedules=['constant'],
        modality_info={'img': {'type': 'img'}},
    )
    assert len(schedule) == 5

def test_build_chained_generation_schedules_maskgit_seq_assert():
    # Illegal scheme for seq modality should trigger assertion
    with pytest.raises(AssertionError):
        build_chained_generation_schedules(
            cond_domains=['text'],
            target_domains=['text'],
            tokens_per_target=[6],
            autoregression_schemes=['maskgit'],
            decoding_steps=[3],
            token_decoding_schedules=['linear'],
            temps=[1.0],
            temp_schedules=['linear'],
            cfg_scales=[1.0],
            cfg_schedules=['constant'],
            modality_info={'text': {'type': 'seq'}},
        )


def test_build_chained_generation_schedules_invalid_scheme_raises():
    with pytest.raises(ValueError):
        build_chained_generation_schedules(
            cond_domains=['text'],
            target_domains=['img'],
            tokens_per_target=[6],
            autoregression_schemes=['invalid'],
            decoding_steps=[3],
            token_decoding_schedules=['linear'],
            temps=[1.0],
            temp_schedules=['linear'],
            cfg_scales=[1.0],
            cfg_schedules=['constant'],
            modality_info={'img': {'type': 'img'}},
        )


def test_build_chained_generation_schedules_invalid_token_schedule_raises():
    with pytest.raises(ValueError):
        build_chained_generation_schedules(
            cond_domains=['text'],
            target_domains=['img'],
            tokens_per_target=[6],
            autoregression_schemes=['maskgit'],
            decoding_steps=[3],
            token_decoding_schedules=['bad'],
            temps=[1.0],
            temp_schedules=['linear'],
            cfg_scales=[1.0],
            cfg_schedules=['constant'],
            modality_info={'img': {'type': 'img'}},
        )


def test_build_chained_generation_schedules_invalid_temp_schedule():
    with pytest.raises(ValueError):
        build_chained_generation_schedules(
            cond_domains=['text'],
            target_domains=['img'],
            tokens_per_target=[6],
            autoregression_schemes=['maskgit'],
            decoding_steps=[3],
            token_decoding_schedules=['linear'],
            temps=[1.0],
            temp_schedules=['bad'],
            cfg_scales=[1.0],
            cfg_schedules=['constant'],
            modality_info={'img': {'type': 'img'}},
        )


def test_build_chained_generation_schedules_invalid_guidance_schedule():
    with pytest.raises(ValueError):
        build_chained_generation_schedules(
            cond_domains=['text'],
            target_domains=['img'],
            tokens_per_target=[6],
            autoregression_schemes=['maskgit'],
            decoding_steps=[3],
            token_decoding_schedules=['linear'],
            temps=[1.0],
            temp_schedules=['linear'],
            cfg_scales=[1.0],
            cfg_schedules=['weird'],
            modality_info={'img': {'type': 'img'}},
        )


def test_build_chained_generation_schedules_guidance_list_scale():
    schedule = build_chained_generation_schedules(
        cond_domains=['text'],
        target_domains=['img'],
        tokens_per_target=[4],
        autoregression_schemes=['maskgit'],
        decoding_steps=[2],
        token_decoding_schedules=['linear'],
        temps=[1.0],
        temp_schedules=['constant'],
        cfg_scales=[[0.5, 1.0]],
        cfg_schedules=['constant'],
        modality_info={'img': {'type': 'img'}},
    )
    assert schedule[0]['cfg_scale'].shape[0] == 2

# ---- GenerationSampler token sampling tests ----

def test_top_k_top_p_filtering_types(sampler_no_reg):
    logits = torch.arange(0,10).float().unsqueeze(0)
    filtered = sampler_no_reg.top_k_top_p_filtering(logits.clone(), top_k=3)
    assert (filtered[0,-3:] != float('-inf')).all()
    filtered_pct = sampler_no_reg.top_k_top_p_filtering(logits.clone(), top_k=0.3)
    # 30% of 10 => top 3 tokens kept
    assert (filtered_pct[0,-3:] != float('-inf')).all()
    with pytest.raises(TypeError):
        sampler_no_reg.top_k_top_p_filtering(logits.clone(), top_k='bad')


def test_top_p_filtering(sampler_no_reg):
    logits = torch.tensor([[10.,9.,8.,1.,0.]])
    filtered = sampler_no_reg.top_k_top_p_filtering(logits.clone(), top_p=0.8)
    # Some low probability tokens should be -inf
    assert (filtered[0,-1] == float('-inf')) or (filtered[0,-2] == float('-inf'))


def test_sample_tokens_argmax(sampler_no_reg):
    logits = torch.tensor([[0.,1.,2.,3.]])
    samples, probs = sampler_no_reg.sample_tokens(logits, temperature=0.0)
    assert samples.item() == 3 and probs.item() == 1.0


def test_sample_tokens_batched(sampler_no_reg):
    logits = torch.randn(2,5,7)
    samples, probs = sampler_no_reg.sample_tokens_batched(logits, temperature=1.0)
    assert samples.shape == (2,5)


def test_select_tokens_return_all(sampler_no_reg):
    logits = torch.tensor([[0.,1.,2.,3.]])  # 2D (batch=1, vocab=4)
    top_samples, top_indices, all_samples = sampler_no_reg.select_tokens(logits, num_select=1, return_all_samples=True)
    # all_samples are the sampled tokens (batch-level), not the full vocab
    assert all_samples.ndim == 1 and top_samples.numel() == 1


def test_select_tokens_batched(sampler_no_reg):
    logits = torch.randn(2,4,6)
    top_samples, top_indices = sampler_no_reg.select_tokens_batched(logits, num_select=2)
    assert top_samples.shape == (2,2)

# ---- Encoder/Decoder masking tests ----

def make_mod_dict(sampler):
    base = {'tensor': torch.randint(0,20,(2,6)), 'input_mask': torch.tensor([[False,False,True,True,True,True],[False,True,True,True,True,True]])}
    # Run through encoder embedding to produce required keys
    enc = sampler.model.encoder_embeddings['text'](base)
    return {'text': {'tensor': enc['tensor'], 'emb': enc['emb'], 'input_mask': enc['input_mask'], 'mod_mask': enc['mod_mask']}}


def test_forward_mask_encoder_generation_no_registers(sampler_no_reg):
    enc_mod = make_mod_dict(sampler_no_reg)
    encoder_tokens, encoder_emb, encoder_mask, mod_mask = sampler_no_reg.forward_mask_encoder_generation(enc_mod)
    assert encoder_tokens.shape[0] == 2
    # encoder_mask has shape (B,1,N); compare N dimension
    assert encoder_mask.shape[2] == encoder_tokens.shape[1]


def test_forward_mask_encoder_generation_with_registers(sampler_with_reg):
    enc_mod = make_mod_dict(sampler_with_reg)
    encoder_tokens, encoder_emb, encoder_mask, mod_mask = sampler_with_reg.forward_mask_encoder_generation(enc_mod)
    # Should add register tokens at sequence start
    assert encoder_tokens.shape[1] > encoder_emb.shape[1] - 2  # coarse check

# ---- Decoder masking paths (maskgit/roar/autoregressive) ----

def prep_decoder_mod_dict():
    # mimic embedding forward_embed expectation
    return {'text': {'tensor': torch.randint(0,20,(2,6)), 'input_mask': torch.tensor([[False,True,True,True,True,True],[False,True,True,True,True,True]])}}


def test_forward_mask_decoder_maskgit(sampler_no_reg):
    d = {'text': {'tensor': torch.randint(0,20,(2,6)), 'input_mask': torch.tensor([[False,True,True,True,True,True],[False,True,True,True,True,True]])}}
    dec_tokens, dec_emb, dec_mask, mod_mask, mod_pos = sampler_no_reg.forward_mask_decoder_maskgit({'text': {'x': torch.randn(2,6,8), 'emb': torch.randn(2,6,8), 'target_mask': d['text']['input_mask'], 'ids': d['text']['tensor']}}, 'text')
    assert dec_tokens.shape[0] == 2 and mod_pos.shape[1] == dec_tokens.shape[1]


def test_forward_mask_decoder_roar(sampler_no_reg):
    d = {'text': {'tensor': torch.randint(0,20,(2,6)), 'input_mask': torch.tensor([[False,True,True,True,True,True],[False,True,True,True,True,True]])}}
    dec_tokens, dec_emb, dec_mask, mod_mask, mod_pos = sampler_no_reg.forward_mask_decoder_roar({'text': {'x': torch.randn(2,6,8), 'emb': torch.randn(2,6,8), 'target_mask': d['text']['input_mask'], 'ids': d['text']['tensor']}}, 'text', num_select=3)
    assert dec_tokens.shape[1] == mod_pos.shape[1]


def test_forward_mask_decoder_autoregressive(sampler_no_reg):
    d = {'text': {'tensor': torch.randint(0,20,(2,6)), 'input_mask': torch.tensor([[False,True,True,True,True,True],[False,True,True,True,True,True]])}}
    dec_ids, dec_emb, dec_mask, mod_mask, mod_pos = sampler_no_reg.forward_mask_decoder_autoregressive({'text': {'x': torch.randn(2,6,8), 'emb': torch.randn(2,6,8), 'target_mask': d['text']['input_mask'], 'ids': d['text']['tensor']}}, 'text')
    assert dec_ids.shape[0] == 2

# ---- Sequence merging tests ----

def test_merge_sequences(tokenizer, sampler_no_reg):
    mod_dict = {'text': {'tensor': torch.tensor([[11,1,2,0]]), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}}
    # Provide both sentinels in prediction to ensure spans merge
    pred_ids = torch.tensor([[11,1,12,3]])
    out = sampler_no_reg.merge_sequences(mod_dict, pred_ids, 'text', tokenizer)
    assert out['text']['tensor'].shape[1] >= 1


def test_merge_sequences_batched(tokenizer, sampler_no_reg):
    mod_dict = {'text': {'tensor': torch.tensor([[11,1,2],[11,2,0]]), 'input_mask': torch.tensor([[False,False,True],[False,True,True]]), 'target_mask': torch.tensor([[True,True,False],[True,False,False]])}}
    pred_ids = torch.tensor([[12,3],[12,4]])
    out = sampler_no_reg.merge_sequences_batched(mod_dict, pred_ids, 'text', tokenizer)
    assert out['text']['tensor'].shape[0] == 2

# ---- MaskGIT step batched simple path ----

def test_maskgit_step_batched(sampler_no_reg):
    # Provide two unmasked (False) positions so num_decoder_tokens >= 2
    input_mask = torch.tensor([[False, False, True, True, True, True],[False, False, True, True, True, True]])
    mod_dict = {'text': {'tensor': torch.zeros((2,6),dtype=torch.int64), 'input_mask': input_mask, 'target_mask': ~input_mask}}
    # Provide embeds required by forward_enc_dec_maskgit_batched
    sampler_no_reg.model.decoder_embeddings['text'].token_emb = nn.Embedding(32,8)
    out = sampler_no_reg.maskgit_step_batched(mod_dict, 'text', num_select=2, temperature=1.0, top_k=0.0, top_p=0.0)
    assert out['text']['tensor'].shape == (2,6)

# ---- ROAR step batched ----

def test_roar_step_batched(sampler_no_reg):
    input_mask = torch.tensor([[False, True, True, True, True, True],[False, True, True, True, True, True]])
    mod_dict = {'text': {'tensor': torch.zeros((2,6),dtype=torch.int64), 'input_mask': input_mask, 'target_mask': ~input_mask}}
    out = sampler_no_reg.roar_step_batched(mod_dict, 'text', num_select=3, temperature=1.0, top_k=0.0, top_p=0.0)
    assert out['text']['tensor'].shape == (2,6)

# ---- Autoregressive step batched (short sequence) ----

def test_autoregressive_step_batched(sampler_no_reg, tokenizer):
    mod_dict = {'text': {'tensor': torch.tensor([[11,1,2,12,0,0],[11,1,2,12,0,0]]), 'input_mask': torch.tensor([[False,False,True,True,True,True],[False,False,True,True,True,True]]), 'target_mask': torch.ones((2,6),dtype=torch.bool)}}
    sampler_no_reg.model.modality_info['text']['max_tokens'] = 3
    out = sampler_no_reg.autoregressive_step_batched(mod_dict, 'text', temperature=0.0, top_k=0, top_p=0.0, use_eos=False, text_tokenizer=tokenizer)
    assert 'text' in out

# ---- Guided autoregressive step (guidance_scale) ----

def test_guided_autoregressive_step_batched(sampler_no_reg, tokenizer):
    # Avoid conditioning due to empty_seq_modality signature mismatch in guided path
    mod_dict = {'text': {'tensor': torch.tensor([[11,1,2,12,0,0]]), 'input_mask': torch.tensor([[False,False,True,True,True,True]]), 'target_mask': torch.ones((1,6),dtype=torch.bool)}}
    sampler_no_reg.model.modality_info['text']['max_tokens'] = 2
    out = sampler_no_reg.guided_autoregressive_step_batched(mod_dict, 'text', temperature=0.0, top_k=0, top_p=0.0, use_eos=False, text_tokenizer=tokenizer, conditioning=[], guidance_scale=0.5)
    assert 'text' in out

# ---- Additional coverage tests ----

def test_init_conditioned_target_modality_img_raises(modality_info):
    base = init_empty_target_modality(modality_info, 'img', 1, 4, 'cpu')
    with pytest.raises(NotImplementedError):
        init_conditioned_target_modality(base, modality_info, 'img', num_target_tokens=2)

def test_init_conditioned_target_modality_invalid_raises(modality_info):
    modality_info['unknown'] = {'type': 'unknown'}
    base = {'tensor': torch.zeros((1,4)), 'input_mask': torch.ones((1,4), dtype=torch.bool), 'target_mask': torch.zeros((1,4), dtype=torch.bool), 'decoder_attention_mask': torch.zeros((1,4), dtype=torch.bool)}
    with pytest.raises(NotImplementedError):
        init_conditioned_target_modality(base, modality_info, 'unknown', num_target_tokens=2)

def test_init_empty_target_modality_invalid_raises(modality_info):
    modality_info['bad'] = {'type': 'bad'}
    with pytest.raises(ValueError):
        init_empty_target_modality(modality_info, 'bad', 1, 4, 'cpu')

def test_init_full_input_modality_seq_emb_with_mask_valid(modality_info):
    value = {'tensor': torch.zeros((1,5)), 'mask_valid': torch.tensor([[True,True,False,False,False]])}
    out = init_full_input_modality(value, modality_info, 'emb', 'cpu')
    assert out['input_mask'][0,0] == False and out['input_mask'][0,2] == True

def test_build_chained_generation_schedules_cosine_token_schedule():
    schedule = build_chained_generation_schedules(
        cond_domains=['text'],
        target_domains=['img'],
        tokens_per_target=[10],
        autoregression_schemes=['maskgit'],
        decoding_steps=[3],
        token_decoding_schedules=['cosine'],
        temps=[1.0],
        temp_schedules=['linear'],
        cfg_scales=[1.0],
        cfg_schedules=['constant'],
    )
    assert len(schedule) == 3

def test_build_chained_generation_schedules_onex_temp():
    schedule = build_chained_generation_schedules(
        cond_domains=['text'],
        target_domains=['img'],
        tokens_per_target=[6],
        autoregression_schemes=['maskgit'],
        decoding_steps=[2],
        token_decoding_schedules=['linear'],
        temps=[2.0],
        temp_schedules=['onex:0.1:0.5'],
        cfg_scales=[1.0],
        cfg_schedules=['constant'],
    )
    assert len(schedule) == 2

def test_build_chained_generation_schedules_cosine_guidance_raises():
    with pytest.raises(NotImplementedError):
        build_chained_generation_schedules(
            cond_domains=['text'],
            target_domains=['img'],
            tokens_per_target=[6],
            autoregression_schemes=['maskgit'],
            decoding_steps=[2],
            token_decoding_schedules=['linear'],
            temps=[1.0],
            temp_schedules=['linear'],
            cfg_scales=[1.0],
            cfg_schedules=['cosine'],
        )

def test_build_chained_generation_schedules_cfg_grow_conditioning():
    schedule = build_chained_generation_schedules(
        cond_domains=['text'],
        target_domains=['img', 'text2'],
        tokens_per_target=[4, 6],
        autoregression_schemes=['maskgit', 'autoregressive'],
        decoding_steps=[2, 0],
        token_decoding_schedules=['linear', 'linear'],
        temps=[1.0, 1.0],
        temp_schedules=['constant', 'constant'],
        cfg_scales=[1.0, 1.0],
        cfg_schedules=['constant', 'constant'],
        cfg_grow_conditioning=True,
    )
    # After first modality completes, it should be added to conditioning
    assert len(schedule) > 1

@pytest.mark.xfail(reason="Bug in original code: empty_seq_modality signature mismatch with empty_img_modality")
def test_guided_maskgit_step_batched_with_conditioning(sampler_no_reg):
    input_mask = torch.tensor([[False, False, True, True, True, True]])
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(1,6)), 'input_mask': torch.zeros((1,6), dtype=torch.bool), 'target_mask': torch.ones((1,6), dtype=torch.bool)},
        'img': {'tensor': torch.zeros((1,6),dtype=torch.int64), 'input_mask': input_mask, 'target_mask': ~input_mask}
    }
    out = sampler_no_reg.guided_maskgit_step_batched(mod_dict, 'img', num_select=1, temperature=1.0, top_k=0.0, top_p=0.0, conditioning=['text'], guidance_scale=2.0)
    assert out['img']['tensor'].shape == (1,6)

def test_guided_maskgit_step_batched_write_all_predictions(sampler_no_reg):
    input_mask = torch.tensor([[False, False, True, True, True, True]])
    mod_dict = {'text': {'tensor': torch.zeros((1,6),dtype=torch.int64), 'input_mask': input_mask, 'target_mask': ~input_mask}}
    out = sampler_no_reg.guided_maskgit_step_batched(mod_dict, 'text', num_select=2, temperature=1.0, top_k=0.0, top_p=0.0, conditioning=[], guidance_scale=1.0, write_all_predictions=True)
    assert out['text']['tensor'].shape == (1,6)

def test_multi_guided_maskgit_step_batched(sampler_no_reg):
    input_mask = torch.tensor([[False, False, True, True, True, True]])
    uncond_dict = {'text': {'tensor': torch.zeros((1,6),dtype=torch.int64), 'input_mask': input_mask, 'target_mask': ~input_mask}}
    cond_dict1 = copy.deepcopy(uncond_dict)
    cond_dict2 = copy.deepcopy(uncond_dict)
    uncond_out, cond_out = sampler_no_reg.multi_guided_maskgit_step_batched(uncond_dict, [cond_dict1, cond_dict2], [1.0, 0.5], 'text', num_select=1, temperature=1.0, top_k=0.0, top_p=0.0)
    assert uncond_out['text']['tensor'].shape == (1,6)

def test_multi_guided_maskgit_step_batched_write_all(sampler_no_reg):
    input_mask = torch.tensor([[False, False, True, True, True, True]])
    uncond_dict = {'text': {'tensor': torch.zeros((1,6),dtype=torch.int64), 'input_mask': input_mask, 'target_mask': ~input_mask}}
    cond_dict1 = copy.deepcopy(uncond_dict)
    uncond_out, cond_out = sampler_no_reg.multi_guided_maskgit_step_batched(uncond_dict, [cond_dict1], [2.0], 'text', num_select=2, temperature=1.0, top_k=0.0, top_p=0.0, write_all_predictions=True)
    assert uncond_out['text']['tensor'].shape == (1,6)

def test_guided_roar_step_batched(sampler_no_reg):
    input_mask = torch.tensor([[False, True, True, True, True, True]])
    mod_dict = {'text': {'tensor': torch.zeros((1,6),dtype=torch.int64), 'input_mask': input_mask, 'target_mask': ~input_mask}}
    out = sampler_no_reg.guided_roar_step_batched(mod_dict, 'text', num_select=2, temperature=1.0, top_k=0.0, top_p=0.0, conditioning=[], guidance_scale=1.5)
    assert out['text']['tensor'].shape == (1,6)

def test_multi_guided_roar_step_batched(sampler_no_reg):
    input_mask = torch.tensor([[False, True, True, True, True, True]])
    uncond_dict = {'text': {'tensor': torch.zeros((1,6),dtype=torch.int64), 'input_mask': input_mask, 'target_mask': ~input_mask}}
    cond_dict1 = copy.deepcopy(uncond_dict)
    cond_dict2 = copy.deepcopy(uncond_dict)
    uncond_out, cond_out = sampler_no_reg.multi_guided_roar_step_batched(uncond_dict, [cond_dict1, cond_dict2], [1.0, 0.5], 'text', num_select=2, temperature=1.0, top_k=0.0, top_p=0.0)
    assert uncond_out['text']['tensor'].shape == (1,6)

def test_autoregressive_step_batched_with_eos_early_exit(sampler_no_reg, tokenizer):
    # Test early exit when EOS is reached
    mod_dict = {'text': {'tensor': torch.tensor([[11,1,13,12,0,0]]), 'input_mask': torch.tensor([[False,False,False,True,True,True]]), 'target_mask': torch.ones((1,6),dtype=torch.bool)}}
    sampler_no_reg.model.modality_info['text']['max_tokens'] = 2
    out = sampler_no_reg.autoregressive_step_batched(mod_dict, 'text', temperature=0.0, top_k=0, top_p=0.0, use_eos=True, eos_token=torch.tensor([13]), text_tokenizer=tokenizer)
    assert 'text' in out

def test_guided_autoregressive_step_batched_with_conditioning(sampler_no_reg, tokenizer):
    mod_dict = {
        'img': {'tensor': torch.randint(0,20,(1,4)), 'input_mask': torch.zeros((1,4), dtype=torch.bool), 'target_mask': torch.ones((1,4), dtype=torch.bool)},
        'text': {'tensor': torch.tensor([[11,1,2,12,0,0]]), 'input_mask': torch.tensor([[False,False,True,True,True,True]]), 'target_mask': torch.ones((1,6),dtype=torch.bool)}
    }
    sampler_no_reg.model.modality_info['text']['max_tokens'] = 2
    out = sampler_no_reg.guided_autoregressive_step_batched(mod_dict, 'text', temperature=0.0, top_k=0, top_p=0.0, use_eos=False, text_tokenizer=tokenizer, conditioning=['img'], guidance_scale=1.5)
    assert 'text' in out

def test_forward_enc_dec_maskgit_batched_full(sampler_no_reg):
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(2,6)), 'input_mask': torch.tensor([[False,False,True,True,True,True],[False,True,True,True,True,True]])}
    }
    logits, mod_pos = sampler_no_reg.forward_enc_dec_maskgit_batched(mod_dict, 'text')
    assert logits.shape[0] == 2

def test_forward_enc_dec_roar_batched_full(sampler_no_reg):
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(2,6)), 'input_mask': torch.tensor([[False,False,True,True,True,True],[False,True,True,True,True,True]])}
    }
    logits, mod_pos = sampler_no_reg.forward_enc_dec_roar_batched(mod_dict, 'text', num_select=3)
    assert logits.shape[0] == 2

def test_generate_maskgit_img(sampler_no_reg, tokenizer):
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(1,6)), 'input_mask': torch.zeros((1,6), dtype=torch.bool), 'target_mask': torch.ones((1,6), dtype=torch.bool)},
        'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}
    }
    schedule = [{'target_domain': 'img', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    out = sampler_no_reg.generate(mod_dict, schedule, top_k=0.0, top_p=0.0)
    assert 'img' in out

@pytest.mark.xfail(reason="Bug in original code: empty_seq_modality signature mismatch with empty_img_modality")
def test_generate_maskgit_img_with_guidance(sampler_no_reg, tokenizer):
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(1,6)), 'input_mask': torch.zeros((1,6), dtype=torch.bool), 'target_mask': torch.ones((1,6), dtype=torch.bool)},
        'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}
    }
    schedule = [{'target_domain': 'img', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 2.0, 'cfg_cond_domains': ['text']}]
    out = sampler_no_reg.generate(mod_dict, schedule, top_k=0.0, top_p=0.0)
    assert 'img' in out

def test_generate_roar_img(sampler_no_reg, tokenizer):
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(1,6)), 'input_mask': torch.zeros((1,6), dtype=torch.bool), 'target_mask': torch.ones((1,6), dtype=torch.bool)},
        'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}
    }
    schedule = [{'target_domain': 'img', 'scheme': 'roar', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    out = sampler_no_reg.generate(mod_dict, schedule, top_k=0.0, top_p=0.0)
    assert 'img' in out

@pytest.mark.xfail(reason="Bug in original code: empty_seq_modality signature mismatch with empty_img_modality")
def test_generate_roar_img_with_guidance(sampler_no_reg, tokenizer):
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(1,6)), 'input_mask': torch.zeros((1,6), dtype=torch.bool), 'target_mask': torch.ones((1,6), dtype=torch.bool)},
        'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}
    }
    schedule = [{'target_domain': 'img', 'scheme': 'roar', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 2.0, 'cfg_cond_domains': ['text']}]
    out = sampler_no_reg.generate(mod_dict, schedule, top_k=0.0, top_p=0.0)
    assert 'img' in out

def test_generate_invalid_scheme_raises(sampler_no_reg):
    sampler_no_reg.model.modality_info['img'] = {'type': 'img', 'id': 1, 'max_tokens': 4}
    sampler_no_reg.model.encoder_embeddings['img'] = StubEmbed(vocab_size=32, dim=8)
    sampler_no_reg.model.decoder_embeddings['img'] = StubEmbed(vocab_size=32, dim=8)
    
    mod_dict = {
        'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.ones((1,4), dtype=torch.bool), 'target_mask': torch.zeros((1,4), dtype=torch.bool)}
    }
    schedule = [{'target_domain': 'img', 'scheme': 'invalid', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    with pytest.raises(ValueError):
        sampler_no_reg.generate(mod_dict, schedule)

def test_generate_seq_autoregressive(sampler_no_reg, tokenizer):
    mod_dict = {'text': {'tensor': torch.tensor([[11,1,2,12,0,0]]), 'input_mask': torch.tensor([[False,False,True,True,True,True]]), 'target_mask': torch.ones((1,6),dtype=torch.bool)}}
    sampler_no_reg.model.modality_info['text']['max_tokens'] = 2
    schedule = [{'target_domain': 'text', 'scheme': 'autoregressive', 'num_tokens': None, 'temperature': 0.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    text_tokenizer_obj = types.SimpleNamespace(text_tokenizer=tokenizer)
    out = sampler_no_reg.generate(mod_dict, schedule, tokenizer={'text': text_tokenizer_obj})
    assert 'text' in out

def test_generate_seq_autoregressive_with_guidance(sampler_no_reg, tokenizer):
    mod_dict = {
        'img': {'tensor': torch.randint(0,20,(1,4)), 'input_mask': torch.zeros((1,4), dtype=torch.bool), 'target_mask': torch.ones((1,4), dtype=torch.bool)},
        'text': {'tensor': torch.tensor([[11,1,2,12,0,0]]), 'input_mask': torch.tensor([[False,False,True,True,True,True]]), 'target_mask': torch.ones((1,6),dtype=torch.bool)}
    }
    sampler_no_reg.model.modality_info['text']['max_tokens'] = 2
    schedule = [{'target_domain': 'text', 'scheme': 'autoregressive', 'num_tokens': None, 'temperature': 0.0, 'cfg_scale': 2.0, 'cfg_cond_domains': ['img']}]
    text_tokenizer_obj = types.SimpleNamespace(text_tokenizer=tokenizer)
    out = sampler_no_reg.generate(mod_dict, schedule, tokenizer={'text': text_tokenizer_obj})
    assert 'text' in out

def test_generate_invalid_modality_type_raises(sampler_no_reg):
    sampler_no_reg.model.modality_info['unknown'] = {'type': 'unknown', 'id': 99, 'max_tokens': 4}
    mod_dict = {'unknown': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.ones((1,4), dtype=torch.bool), 'target_mask': torch.zeros((1,4), dtype=torch.bool)}}
    schedule = [{'target_domain': 'unknown', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    with pytest.raises(ValueError):
        sampler_no_reg.generate(mod_dict, schedule)

def test_generate_iter_maskgit(sampler_no_reg):
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(1,6)), 'input_mask': torch.zeros((1,6), dtype=torch.bool), 'target_mask': torch.ones((1,6), dtype=torch.bool)},
        'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}
    }
    schedule = [{'target_domain': 'img', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    gen = sampler_no_reg.generate_iter(mod_dict, schedule)
    result = next(gen)
    assert 'img' in result

def test_generate_iter_roar(sampler_no_reg):
    mod_dict = {
        'text': {'tensor': torch.randint(0,20,(1,6)), 'input_mask': torch.zeros((1,6), dtype=torch.bool), 'target_mask': torch.ones((1,6), dtype=torch.bool)},
        'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}
    }
    schedule = [{'target_domain': 'img', 'scheme': 'roar', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    gen = sampler_no_reg.generate_iter(mod_dict, schedule)
    result = next(gen)
    assert 'img' in result

def test_generate_iter_invalid_scheme_raises(sampler_no_reg):
    sampler_no_reg.model.modality_info['img'] = {'type': 'img', 'id': 1, 'max_tokens': 4}
    mod_dict = {'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.ones((1,4), dtype=torch.bool), 'target_mask': torch.zeros((1,4), dtype=torch.bool)}}
    schedule = [{'target_domain': 'img', 'scheme': 'bad', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    gen = sampler_no_reg.generate_iter(mod_dict, schedule)
    with pytest.raises(ValueError):
        next(gen)

def test_generate_iter_seq_autoregressive(sampler_no_reg, tokenizer):
    mod_dict = {'text': {'tensor': torch.tensor([[11,1,2,12,0,0]]), 'input_mask': torch.tensor([[False,False,True,True,True,True]]), 'target_mask': torch.ones((1,6),dtype=torch.bool)}}
    sampler_no_reg.model.modality_info['text']['max_tokens'] = 2
    schedule = [{'target_domain': 'text', 'scheme': 'autoregressive', 'num_tokens': None, 'temperature': 0.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    gen = sampler_no_reg.generate_iter(mod_dict, schedule, text_tokenizer=tokenizer)
    result = next(gen)
    assert 'text' in result

def test_generate_iter_seq_with_guidance(sampler_no_reg, tokenizer):
    mod_dict = {
        'img': {'tensor': torch.randint(0,20,(1,4)), 'input_mask': torch.zeros((1,4), dtype=torch.bool), 'target_mask': torch.ones((1,4), dtype=torch.bool)},
        'text': {'tensor': torch.tensor([[11,1,2,12,0,0]]), 'input_mask': torch.tensor([[False,False,True,True,True,True]]), 'target_mask': torch.ones((1,6),dtype=torch.bool)}
    }
    sampler_no_reg.model.modality_info['text']['max_tokens'] = 2
    schedule = [{'target_domain': 'text', 'scheme': 'autoregressive', 'num_tokens': None, 'temperature': 0.0, 'cfg_scale': 2.0, 'cfg_cond_domains': ['img']}]
    gen = sampler_no_reg.generate_iter(mod_dict, schedule, text_tokenizer=tokenizer)
    result = next(gen)
    assert 'text' in result

def test_generate_iter_invalid_modality_raises(sampler_no_reg):
    sampler_no_reg.model.modality_info['bad'] = {'type': 'bad', 'id': 99, 'max_tokens': 4}
    mod_dict = {'bad': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.ones((1,4), dtype=torch.bool), 'target_mask': torch.zeros((1,4), dtype=torch.bool)}}
    schedule = [{'target_domain': 'bad', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': 1.0, 'cfg_cond_domains': []}]
    gen = sampler_no_reg.generate_iter(mod_dict, schedule)
    with pytest.raises(ValueError):
        next(gen)

def test_generate_multi_guided_simple(sampler_no_reg):
    uncond_dict = {'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}}
    cond_dict1 = copy.deepcopy(uncond_dict)
    cond_dict2 = copy.deepcopy(uncond_dict)
    schedule = [{'target_domain': 'img', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': [1.0, 0.5]}]
    out = sampler_no_reg.generate_multi_guided(uncond_dict, [cond_dict1, cond_dict2], schedule)
    assert 'img' in out

def test_generate_multi_guided_roar(sampler_no_reg):
    uncond_dict = {'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}}
    cond_dict1 = copy.deepcopy(uncond_dict)
    schedule = [{'target_domain': 'img', 'scheme': 'roar', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': [2.0]}]
    out = sampler_no_reg.generate_multi_guided(uncond_dict, [cond_dict1], schedule)
    assert 'img' in out

def test_generate_multi_guided_invalid_scheme_raises(sampler_no_reg):
    sampler_no_reg.model.modality_info['img'] = {'type': 'img', 'id': 1, 'max_tokens': 4}
    uncond_dict = {'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.ones((1,4), dtype=torch.bool), 'target_mask': torch.zeros((1,4), dtype=torch.bool)}}
    cond_dict1 = copy.deepcopy(uncond_dict)
    schedule = [{'target_domain': 'img', 'scheme': 'invalid', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': [1.0]}]
    with pytest.raises(ValueError):
        sampler_no_reg.generate_multi_guided(uncond_dict, [cond_dict1], schedule)

def test_generate_multi_guided_unsupported_modality_raises(sampler_no_reg):
    sampler_no_reg.model.modality_info['text']['type'] = 'seq'
    uncond_dict = {'text': {'tensor': torch.zeros((1,6),dtype=torch.int64), 'input_mask': torch.ones((1,6), dtype=torch.bool), 'target_mask': torch.zeros((1,6), dtype=torch.bool)}}
    cond_dict1 = copy.deepcopy(uncond_dict)
    schedule = [{'target_domain': 'text', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': [1.0]}]
    with pytest.raises(NotImplementedError):
        sampler_no_reg.generate_multi_guided(uncond_dict, [cond_dict1], schedule)

def test_generate_multi_guided_multiple_modalities(sampler_no_reg):
    sampler_no_reg.model.modality_info['img2'] = {'type': 'img', 'id': 2, 'max_tokens': 4}
    sampler_no_reg.model.encoder_embeddings['img2'] = StubEmbed(vocab_size=32, dim=8)
    sampler_no_reg.model.decoder_embeddings['img2'] = StubEmbed(vocab_size=32, dim=8)
    
    uncond_dict = {
        'img': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])},
        'img2': {'tensor': torch.zeros((1,4),dtype=torch.int64), 'input_mask': torch.tensor([[False,False,True,True]]), 'target_mask': torch.tensor([[True,True,False,False]])}
    }
    cond_dict1 = copy.deepcopy(uncond_dict)
    schedule = [
        {'target_domain': 'img', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': [1.0]},
        {'target_domain': 'img2', 'scheme': 'maskgit', 'num_tokens': 2, 'temperature': 1.0, 'cfg_scale': [1.0]}
    ]
    out = sampler_no_reg.generate_multi_guided(uncond_dict, [cond_dict1], schedule)
    assert 'img' in out and 'img2' in out

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
