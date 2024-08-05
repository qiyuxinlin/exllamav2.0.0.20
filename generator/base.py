from __future__ import annotations

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2CacheBase,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2Sampler
)
import torch
import random
import math
import threading
from exllamav2.generator.hooks import ExLlamaV2PostSamplingHook, ExLlamaV2PostSamplingResult
from exllamav2.embedding import EMBEDDING_INDEX
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from scipy import signal
import os
import re
class ExLlamaV2BaseGenerator:

    # Internal state

    model: ExLlamaV2
    cache: ExLlamaV2CacheBase
    tokenizer: ExLlamaV2Tokenizer

    sequence_ids: torch.Tensor | None

    abort_event: threading.Event | None


    def __init__(self,
                 model: ExLlamaV2,
                 cache: ExLlamaV2CacheBase,
                 tokenizer: ExLlamaV2Tokenizer):

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.sequence_ids = None
        self.abort_event = None

    # For testing purposes, run a forward pass to make sure CUDA is fully initialized

    def warmup(self):

        input_ids = torch.zeros((1, 2), dtype = torch.long)
        self.model.forward(input_ids, cache = None, input_mask = None, preprocess_only = True)
        torch.cuda.synchronize()


    def full(self):

        return self.sequence_ids.shape[-1] >= self.model.config.max_seq_len
    # 从begin_stream_ex薅过来的
    def set_stop_conditions(self,
                            stop_conditions: list | tuple | set):
        """
        :param stop_conditions:
            List of either token IDs or strings that will cause stream_ex to emit the EOS signal. String values do not
            have to match whole tokens and can span multiple tokens.

        Example:
            generator.set_stop_conditions(tokenizer.eos_token_id, "\nUser:", "###")
        """

        self.stop_strings = set()
        self.stop_tokens = set()
        for t in stop_conditions:
            if isinstance(t, int): self.stop_tokens.add(t)
            elif isinstance(t, str): self.stop_strings.add(t)
            else: raise ValueError("Unsupported type in stop_conditions")

    def generate_simple(self,
                        prompt: str or list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed: int or None = None,
                        token_healing: bool = False,
                        encode_special_tokens: bool = False,
                        decode_special_tokens: bool = False,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] | None = None,
                        stop_token: int or None = -1,
                        add_bos: bool = False,
                        abort_event: threading.Event | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        completion_only: bool = False,
                        use_nbce: bool = False):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """


        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Accept LoRA or list of LoRAs

        if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed, inserting embeddings if provided

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        prompts_identical = batch_size == 1 or all(s == prompt[0] for s in prompt)

        if input_embeddings is not None:

            embed_marker = "{{EMBED_HERE}}"
            prompt_split = prompt.split(embed_marker)
            assert len(prompt_split) == 2, \
                f"Prompt must contain one instance of {embed_marker} when embeddings are provided"

            if batch_size > 1: assert prompts_identical, \
                "Batched generation with input embeddings requires all prompts to be identical."

            assert input_embeddings.shape[0] == batch_size, \
                "Input embeddings tensor does not match batch size of prompt."

            pre_ids, _ = self.tokenizer.encode(prompt_split[0].rstrip(" \t"),
                                               encode_special_tokens = encode_special_tokens,
                                               return_offsets = True,
                                               add_bos = add_bos)
            post_ids, _ = self.tokenizer.encode(prompt_split[1].lstrip(" \t"),
                                               encode_special_tokens = encode_special_tokens,
                                               return_offsets = True,
                                               add_bos = False)

            num_emb_tokens = input_embeddings.shape[1]
            image_ids = torch.arange(EMBEDDING_INDEX, EMBEDDING_INDEX + num_emb_tokens, dtype = torch.long).unsqueeze(0)
            ids = torch.cat((pre_ids, image_ids, post_ids), dim = -1)

            position_offsets = None

        else:
            ids, position_offsets = self.tokenizer.encode(prompt,
                                                          encode_special_tokens = encode_special_tokens,
                                                          return_offsets = True,
                                                          add_bos = add_bos)

            if prompts_identical:
                position_offsets = None

        # Truncate prompt if generation would cause cache overflow

        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]
        else: overflow = 0

        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None

        first_token = max(-overflow, 0)

        # Completion only

        if completion_only:
            first_token = ids.shape[-1]

        # Prepare for healing

        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]

        # Process prompt and begin gen

        self._gen_begin_base(ids,
                             mask,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings)

        if self.abort_event and self.abort_event.is_set():
            if isinstance(prompt, str): return ""
            else: return [""] * len(prompt)

        # Remove indexed embeddings from generator's sequence

        if input_embeddings is not None:
            self.sequence_ids[self.sequence_ids >= EMBEDDING_INDEX] = self.tokenizer.pad_token_id

        # Begin filters

        healed_token = []
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None
        gen_settings.begin_filters(heal)

        # Generate tokens

        batch_eos = [False] * batch_size

        for i in range(num_tokens):

            if self.abort_event and self.abort_event.is_set():
                break

            logits = self.model.forward(self.sequence_ids[:, -1:],
                                        self.cache,
                                        input_mask = mask,
                                        loras = loras,
                                        position_offsets = position_offsets,
                                        indexed_embeddings = input_embeddings).float().cpu()
            if use_nbce:
                token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.nbce_sample(logits,
                                                                        gen_settings,
                                                                        self.sequence_ids,
                                                                        random.random(),
                                                                        self.tokenizer,
                                                                        prefix_token = unhealed_token,
                                                                        stop_token = stop_token)
            else:
                token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.sample(logits,
                                                                        gen_settings,
                                                                        self.sequence_ids,
                                                                        random.random(),
                                                                        self.tokenizer,
                                                                        prefix_token = unhealed_token)

            if unhealed_token is not None:
                unhealed_token_copy = unhealed_token
                healed_token = token

            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            # Post sampling hook

            if gen_settings.post_sampling_hooks:
                p = ExLlamaV2PostSamplingResult(
                    sampled_token = token,
                    sampled_prob = prob,
                    logits = logits,
                    candidate_tokens = None if ptokens.is_meta else ptokens,
                    candidate_probs = None if pprobs.is_meta else pprobs
                )
                for h in gen_settings.post_sampling_hooks:
                    h(p)
                token = p.sampled_token
                if p.feed_filters:
                    gen_settings.feed_filters(token)

            else:
                gen_settings.feed_filters(token)

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            unhealed_token = None
            if eos: break

        # Decode

        decode_ids = self.sequence_ids[:, first_token:]
        if input_embeddings is not None:
            decode_ids = torch.stack([decode_ids[i][decode_ids[i] != self.tokenizer.pad_token_id] for i in range(batch_size)])

        if len(healed_token) and completion_only:
            decode_ids = torch.cat([healed_token, decode_ids], dim = -1)

        text = self.tokenizer.decode(decode_ids, decode_special_tokens = decode_special_tokens)

        if len(healed_token) and completion_only:
            pre_text = self.tokenizer.decode(unhealed_token_copy, decode_special_tokens = decode_special_tokens)
            text = [t[len(p):] for t, p in zip(text, pre_text)]

        if isinstance(prompt, str):
            return text[0]
        else:
            return text


    def _gen_begin_base(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.forward(input_ids[:, :-1],
                           self.cache,
                           input_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]

    def _gen_begin_base_with_cache_current_seq_len(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None):

        # self.cache.current_seq_len = 0
        self.sequence_ids = torch.cat((self.sequence_ids,input_ids), dim=1)

        self.model.forward(input_ids[:, :-1],
                           self.cache,
                           input_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]

    def cache_retrival_gen_begin_base(self,
                        input_ids: torch.Tensor,
                        k_need_index:torch.Tensor,
                        tokens_len:list,
                        batch:int,
                        data_name:str,
                        model_name:str,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        rate:float | None = None
                        ):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.cache_retrival_forward(input_ids[:, :-1],
                           self.cache,
                           input_mask = None,
                           preprocess_only = True,
                           loras = None,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings,
                           k_need_index = k_need_index,
                           tokens_len=tokens_len,
                           batch=batch,
                           data_name=data_name,
                           model_name=model_name,
                           rate=rate
                           )
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]


    def cache_query_select_token(self,
                        input_ids: torch.Tensor,
                        importance:torch.Tensor,
                        tokens_len:list,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        ):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.cache_query_select_token(input_ids[:, :-1],
                           self.cache,
                           input_mask = None,
                           preprocess_only = True,
                           loras = None,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings,
                           importance = importance,
                           tokens_len=tokens_len,
                           )
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]

    def _dump_gen_begin_base(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        context_id:int | None = None,
                        index:int | None = None,
                        data_name:str | None = None,
                        model_name:str | None = None,
                        system_len:str | None = None,
                        query_prefix_len:str | None = None,
                        context_num:int | None = None):
        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.dump_forward(input_ids[:, :],
                           self.cache,
                           additional_attn_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings,
                           context_id = context_id,
                           index = index,
                           data_name=data_name,
                           model_name=model_name,
                           system_len=system_len,
                           query_prefix_len=query_prefix_len,
                           context_num=context_num)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]

    def _dump_gen_begin_base_importance(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        context_id:int | None = None,
                        index:int | None = None,
                        data_name:str | None = None,
                        model_name:str | None = None,
                        p_len:int | None = None,
                        system_len:int | None = None):
        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.dump_forward_importance(input_ids[:, :],
                           self.cache,
                           additional_attn_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings,
                           context_id = context_id,
                           index = index,
                           data_name=data_name,
                           model_name=model_name,
                           p_len=p_len,
                           system_len=system_len)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]
    def _gen_begin_base_warm_up(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.one_layer_forward(input_ids[:, :-1],
                           self.cache,
                           additional_attn_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]

    def last_layer_nbce_gen_begin_base(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        whole_mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        last_layer_flag: bool =False,
                        last_layer:int | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.forward(input_ids[:, :-1],
                           self.cache,
                           additional_attn_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings,
                           whole_mask = whole_mask,
                           last_layer_flag=last_layer_flag,
                           last_layer=last_layer)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]

    def cache_blend_warm_up(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        whole_mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        last_layer_flag: bool =False,
                        last_layer:int | None = None,
                        rate:float | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        k_sub_all,v_sub_all,x_sub_all = self.model.cache_blend_forward(input_ids[:, :],
                           self.cache,
                           additional_attn_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings,
                           whole_mask = whole_mask,
                           last_layer_flag=last_layer_flag,
                           last_layer=last_layer,rate=rate)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]
        return k_sub_all,v_sub_all,x_sub_all
    @torch.inference_mode()
    def cache_blend_select_index(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        whole_mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        last_layer_flag: bool =False,
                        last_layer:int | None = None,
                        rate:float | None = None,
                        layer_index:int | None = None,
                        tokens_len:list | None = None,
                        data_name:str | None = None,
                        model_name:str | None = None,
                        batch:int | None = None):
        system_len = tokens_len[0][0]
        constants = self.model.get_device_tensors(0, scratch = False)
        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids
        layer_nums = len(self.cache.key_states)
        for context_id in range(len(tokens_len[0])-1):
            for layer_idx in range(layer_nums):
                batch_keys, batch_values = self.cache.get_kv_state(layer_idx, 0, 0, 0)
                if context_id == 0:
                    cache_key = torch.load(f'./cacheDump/data/{data_name}/{model_name}/{batch}_1_{layer_idx}_key.pt')[:,:system_len]
                    cache_value = torch.load(f'./cacheDump/data/{data_name}/{model_name}/{batch}_1_{layer_idx}_value.pt')[:,:system_len]
                    # ext_c.rope_inverse_(cache_key, constants.sin, constants.cos, 0, cache_key.shape[-2], cache_key.shape[-1], none_tensor, self.model.config.arch.rope_neox_style)
                else:
                    cache_key = torch.load(f'./cacheDump/data/{data_name}/{model_name}/{batch}_{context_id}_{layer_idx}_key.pt')
                    ext_c.rope_inverse_(cache_key, constants.sin, constants.cos, 0, cache_key.shape[-2], cache_key.shape[-1], none_tensor, self.model.config.arch.rope_neox_style)
                    cache_key = cache_key[:,system_len:]
                    ext_c.rope_(cache_key, constants.sin, constants.cos, self.cache.current_seq_len, cache_key.shape[-2], cache_key.shape[-1], none_tensor, self.model.config.arch.rope_neox_style)
                    cache_value = torch.load(f'./cacheDump/data/{data_name}/{model_name}/{batch}_{context_id}_{layer_idx}_value.pt')[:,system_len:]
                new_keys = batch_keys.narrow(1, self.cache.current_seq_len, tokens_len[0][context_id]).narrow(0, 0, 1)
                new_values = batch_values.narrow(1, self.cache.current_seq_len, tokens_len[0][context_id]).narrow(0, 0, 1)
                
                new_keys.copy_(cache_key)
                new_values.copy_(cache_value)
            self.cache.current_seq_len += tokens_len[0][context_id]

        # self.cache.current_seq_len = 0
        # self.model.forward(input_ids[:, :],
        #                    self.cache,
        #                    additional_attn_mask = mask,
        #                    preprocess_only = True,
        #                    loras = loras,
        #                    position_offsets = position_offsets,
        #                    abort_event = self.abort_event,
        #                    indexed_embeddings = input_embeddings,
        #                    whole_mask = whole_mask,
        #                    last_layer_flag=last_layer_flag,
        #                    last_layer=last_layer)
        key_change_mask,value_change_mask = self.cache.get_kv_state(layer_index,0,0,0)
        key_change_mask = key_change_mask[:,:self.cache.current_seq_len,:,:].clone()
        value_change_mask = value_change_mask[:,:self.cache.current_seq_len,:,:].clone()
        self.cache.current_seq_len = 0
        self.model.select_index_forward(input_ids[:, :],
                           self.cache,
                           additional_attn_mask = whole_mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings,
                           whole_mask = whole_mask,
                           last_layer_flag=last_layer_flag,
                           last_layer=last_layer)
        key_full_mask,value_full_mask = self.cache.get_kv_state(layer_index,0,0,0)
        key_full_mask = key_full_mask[:,:self.cache.current_seq_len,:,:].clone()
        value_full_mask = value_full_mask[:,:self.cache.current_seq_len,:,:].clone()
        return key_change_mask, key_full_mask
    
    @torch.inference_mode()
    def cache_recompute_select_index(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        whole_mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        last_layer_flag: bool =False,
                        last_layer:int | None = None,
                        rate:float | None = None,
                        layer_index:int | None = None,
                        tokens_len:list | None = None,
                        data_name:str | None = None,
                        model_name:str | None = None,
                        batch:int | None = None):
        result_change = []
        result_full = []
        system_len = tokens_len[0][0]
        constants = self.model.get_device_tensors(0, scratch = False)
        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids
        layer_nums = len(self.cache.key_states)
        for context_id in range(len(tokens_len[0])-1):
            for layer_idx in range(layer_nums):
                batch_keys, batch_values = self.cache.get_kv_state(layer_idx, 0, 0, 0)
                if context_id == 0:
                    cache_key = torch.load(f'./cacheDump/data/{data_name}/{model_name}/{batch}_1_{layer_idx}_key.pt')[:,:system_len]
                    cache_value = torch.load(f'./cacheDump/data/{data_name}/{model_name}/{batch}_1_{layer_idx}_value.pt')[:,:system_len]
                    # ext_c.rope_inverse_(cache_key, constants.sin, constants.cos, 0, cache_key.shape[-2], cache_key.shape[-1], none_tensor, self.model.config.arch.rope_neox_style)
                else:
                    cache_key = torch.load(f'./cacheDump/data/{data_name}/{model_name}/{batch}_{context_id}_{layer_idx}_key.pt')
                    ext_c.rope_inverse_(cache_key, constants.sin, constants.cos, 0, cache_key.shape[-2], cache_key.shape[-1], none_tensor, self.model.config.arch.rope_neox_style)
                    cache_key = cache_key[:,system_len:]
                    ext_c.rope_(cache_key, constants.sin, constants.cos, self.cache.current_seq_len, cache_key.shape[-2], cache_key.shape[-1], none_tensor, self.model.config.arch.rope_neox_style)
                    cache_value = torch.load(f'./cacheDump/data/{data_name}/{model_name}/{batch}_{context_id}_{layer_idx}_value.pt')[:,system_len:]
                new_keys = batch_keys.narrow(1, self.cache.current_seq_len, tokens_len[0][context_id]).narrow(0, 0, 1)
                new_values = batch_values.narrow(1, self.cache.current_seq_len, tokens_len[0][context_id]).narrow(0, 0, 1)
                
                new_keys.copy_(cache_key)
                new_values.copy_(cache_value)
            self.cache.current_seq_len += tokens_len[0][context_id]

        # self.cache.current_seq_len = 0
        # self.model.forward(input_ids[:, :],
        #                    self.cache,
        #                    additional_attn_mask = mask,
        #                    preprocess_only = True,
        #                    loras = loras,
        #                    position_offsets = position_offsets,
        #                    abort_event = self.abort_event,
        #                    indexed_embeddings = input_embeddings,
        #                    whole_mask = whole_mask,
        #                    last_layer_flag=last_layer_flag,
        #                    last_layer=last_layer)
        for idx in range(1,layer_index+1):
            key_change_mask,value_change_mask = self.cache.get_kv_state(idx,0,0,0)
            key_change_mask = key_change_mask[:,:self.cache.current_seq_len,:,:].clone()
            value_change_mask = value_change_mask[:,:self.cache.current_seq_len,:,:].clone()
            result_change.append(key_change_mask)
        self.cache.current_seq_len = 0
        self.model.select_index_forward(input_ids[:, :],
                           self.cache,
                           additional_attn_mask = whole_mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings,
                           whole_mask = whole_mask,
                           last_layer_flag=last_layer_flag,
                           last_layer=last_layer)
        for idx in range(1,layer_index+1):
            key_full_mask,value_full_mask = self.cache.get_kv_state(idx,0,0,0)
            key_full_mask = key_full_mask[:,:self.cache.current_seq_len,:,:].clone()
            value_full_mask = value_full_mask[:,:self.cache.current_seq_len,:,:].clone()
            result_full.append(key_full_mask)
        return result_change, result_full


    def nbce_gen_begin_base(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.forward(input_ids[:, :-1],
                           self.cache,
                           additional_attn_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]

    def nbce_generate_simple(self,
                        prompt: list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed: int or None = None,
                        token_healing: bool = False,
                        encode_special_tokens: bool = False,
                        decode_special_tokens: bool = False,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] | None = None,
                        stop_token: int or None = -1,
                        add_bos: bool = False,
                        abort_event: threading.Event | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        completion_only: bool = False,
                        mask_flag: bool = False,
                        tail_mask: bool = False,
                        tail_len: int = 50,
                        last_layer_flag: bool = False,
                        last_layer: int | None = None):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """


        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed, inserting embeddings if provided

        batch_size = len(prompt)
        prompts_identical = batch_size == 1 or all(s == prompt[0] for s in prompt)


        # 由于要构造mask，所以要获取每段context的长度，所以得实现一个encode
        tokens_list = []
        tokens_len = []
        for passage in prompt:
            passage_tokens = None
            passage_len = []
            for context in passage:
                context_tokens = self.tokenizer.encode(context, add_bos = add_bos, 
                                                       encode_special_tokens = True)
                passage_len.append(context_tokens.shape[1])
                if passage_tokens is None:
                    passage_tokens = context_tokens
                else:
                    passage_tokens = torch.cat((passage_tokens, context_tokens), dim = 1)

            tokens_list.append(passage_tokens)
            tokens_len.append(passage_len)
        # 每个passage需要补长度
        max_len = max([sum(x) for x in tokens_len])
        for i,len_i in enumerate(tokens_len):
            len_i.insert(0, max_len - sum(len_i))
            padd_tokens = torch.full((1, len_i[0]), self.tokenizer.pad_token_id)
            tokens_list[i] = torch.cat((padd_tokens, tokens_list[i]), dim = 1)
        # 需要返回一个position_offsets tensor
        position_offsets = [offset[0] for offset in tokens_len]
        position_offsets = torch.tensor(position_offsets, dtype = torch.int).unsqueeze(0)
        ids = torch.cat(tokens_list, dim = 0).to(torch.long)
        # text = self.tokenizer.decode(ids[0])
        # print(text)
        if prompts_identical:
            position_offsets = None

        # Truncate prompt if generation would cause cache overflow
        print(f'prompt length: {ids.shape[1]}')
        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]
        else: overflow = 0
        
        # generate阶段传一个全可见的mask回去
        mask_2 = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        # 改mask
        mask_list = []
        sum_len = sum(tokens_len[0])
        for passage_index in range(batch_size):
            inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
            inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
            if passage_index == 0 and batch_size != 1: # pad + query
                current_start = tokens_len[passage_index][1]
                mask = torch.triu(torch.full((current_start, current_start), -65504.0),diagonal=1)
            elif mask_flag == False: # 如果不改mask,走input_masks
                mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
            else: # pad + sys + context + query
                system_len = tokens_len[passage_index][1]
                current_start = system_len
                question_len = tokens_len[passage_index][-1]
                system_context_len = sum(tokens_len[passage_index][1:-1])
                context_len = sum(tokens_len[passage_index][2:-1])
                sys_mask = torch.triu(torch.full((system_len, system_len), -65504.0),diagonal=1)
                context_sys_mask = torch.zeros(context_len, system_len)
                mask = torch.cat((sys_mask,context_sys_mask),dim=0)
                # 拼context
                for length in tokens_len[passage_index][2:-1]:
                    zero_mask = torch.full((current_start,length), -65504.0)
                    context_mask = torch.triu(torch.full((length, length), -65504.0),diagonal=1)
                    current_start += length
                    zero_mask_2 = torch.full((system_context_len - current_start,length), -65504.0)
                    tmp_mask = torch.cat((zero_mask,context_mask,zero_mask_2),dim=0)
                    mask = torch.cat((mask,tmp_mask),dim=1)
                    #尾部填充atten
                if tail_mask == True:
                    current_start = system_len
                    for length in tokens_len[passage_index][2:-1]:
                        current_start += length
                        mask[current_start-tail_len:current_start,system_len:current_start-length] = 0
                # 拼query
                zero_mask_3 = torch.full((system_context_len, question_len), -65504.0)
                mask = torch.cat((mask,zero_mask_3),dim=1)
                question_mask_1 = torch.zeros(question_len, system_context_len)
                question_mask_2 = torch.triu(torch.full((question_len, question_len), -65504.0),diagonal=1)
                question_mask = torch.cat((question_mask_1,question_mask_2),dim=1)
                mask = torch.cat((mask,question_mask),dim=0)

            mask = torch.cat((inf_mask_2,mask),dim=1)
            mask = torch.cat((inf_mask_1,mask),dim=0)
            mask = mask.unsqueeze(0)
            mask_list.append(mask)
        mask = torch.cat(mask_list, dim = 0)

        first_token = max(-overflow, 0)

        if last_layer_flag == True:
            whole_mask_list = []
            for passage_index in range(batch_size):
                inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
                inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
                whole_mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
                whole_mask = torch.cat((inf_mask_2,whole_mask),dim=1)
                whole_mask = torch.cat((inf_mask_1,whole_mask),dim=0)
                whole_mask = whole_mask.unsqueeze(0)
                whole_mask_list.append(whole_mask)
            whole_mask = torch.cat(whole_mask_list, dim = 0)
        
        else:
            whole_mask = None

        # Completion only

        if completion_only:
            first_token = ids.shape[-1]

        # Prepare for healing

        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]

        # Process prompt and begin gen
        # 如果传入的是additional_attn_mask
        if mask.dim() == 3:
            self.last_layer_nbce_gen_begin_base(ids,
                             mask,
                             whole_mask,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings,
                             last_layer_flag=last_layer_flag,
                             last_layer=last_layer)
        # 如果传入的是input_mask
        elif mask.dim() == 2:
            self._gen_begin_base(ids,
                             mask,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            if isinstance(prompt, str): return ""
            else: return [""] * len(prompt)

        # 这个修改对interlm20b来讲，会让模型不废话，效果是在generate之前会把prefill的所有文本都删除掉，对Qwen模型都不好用，会降低模型效果
        # self.sequence_ids = torch.zeros((batch_size, self.sequence_ids.shape[-1]), dtype = torch.int64)
        # Remove indexed embeddings from generator's sequence
        if input_embeddings is not None:
            self.sequence_ids[self.sequence_ids >= EMBEDDING_INDEX] = self.tokenizer.pad_token_id

        # Begin filters

        healed_token = []
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None
        # 这个函数内部是pass
        gen_settings.begin_filters(heal)

        # Generate tokens

        batch_eos = [False] * batch_size
        print('begin generate')
        for i in range(num_tokens):

            if self.abort_event and self.abort_event.is_set():
                break
            # 这里不能传additional_attn_mask，generate阶段要q_len对past_len全可见
            logits = self.model.forward(self.sequence_ids[:, -1:],
                                        self.cache,
                                        input_mask = mask_2,
                                        loras = loras,
                                        position_offsets = position_offsets,
                                        indexed_embeddings = input_embeddings).float().cpu()
            
            token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.nbce_sample(logits,
                                                                        gen_settings,
                                                                        self.sequence_ids,
                                                                        random.random(),
                                                                        self.tokenizer,
                                                                        prefix_token = unhealed_token
                                                                    )

            if unhealed_token is not None:
                unhealed_token_copy = unhealed_token
                healed_token = token

            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            # Post sampling hook

            if gen_settings.post_sampling_hooks:
                p = ExLlamaV2PostSamplingResult(
                    sampled_token = token,
                    sampled_prob = prob,
                    logits = logits,
                    candidate_tokens = None if ptokens.is_meta else ptokens,
                    candidate_probs = None if pprobs.is_meta else pprobs
                )
                for h in gen_settings.post_sampling_hooks:
                    h(p)
                token = p.sampled_token
                if p.feed_filters:
                    gen_settings.feed_filters(token)

            else:# 这个函数下面也是pass
                gen_settings.feed_filters(token)

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            unhealed_token = None
            if batch_size != 1:
                token = token[0]
            if eos or token.item() in self.stop_tokens: break

        # Decode

        decode_ids = self.sequence_ids[:, first_token:]
        if input_embeddings is not None:
            decode_ids = torch.stack([decode_ids[i][decode_ids[i] != self.tokenizer.pad_token_id] for i in range(batch_size)])

        if len(healed_token) and completion_only:
            decode_ids = torch.cat([healed_token, decode_ids], dim = -1)
        if self.tokenizer.decode(decode_ids[0,-1].unsqueeze(0), decode_special_tokens) == '[UNUSED_TOKEN_145]':
            text = self.tokenizer.decode(decode_ids[:,:-1], decode_special_tokens = decode_special_tokens)
        else:
            text = self.tokenizer.decode(decode_ids, decode_special_tokens = decode_special_tokens)
        if len(healed_token) and completion_only:
            pre_text = self.tokenizer.decode(unhealed_token_copy, decode_special_tokens = decode_special_tokens)
            text = [t[len(p):] for t, p in zip(text, pre_text)]

        if isinstance(prompt, str):
            return text[0]
        else:
            return text
        

    def cache_blend_generate_simple(self,
                        prompt: list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed: int or None = None,
                        token_healing: bool = False,
                        encode_special_tokens: bool = False,
                        decode_special_tokens: bool = False,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] | None = None,
                        stop_token: int or None = -1,
                        add_bos: bool = False,
                        abort_event: threading.Event | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        completion_only: bool = False,
                        mask_flag: bool = False,
                        tail_mask: bool = False,
                        tail_len: int = 50,
                        last_layer_flag: bool = False,
                        last_layer: int | None = None,
                        rate: float| None = None):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """


        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed, inserting embeddings if provided

        batch_size = len(prompt)
        prompts_identical = batch_size == 1 or all(s == prompt[0] for s in prompt)


        # 由于要构造mask，所以要获取每段context的长度，所以得实现一个encode
        tokens_list = []
        tokens_len = []
        for passage in prompt:
            passage_tokens = None
            passage_len = []
            for context in passage:
                context_tokens = self.tokenizer.encode(context, add_bos = add_bos, 
                                                       encode_special_tokens = encode_special_tokens)
                passage_len.append(context_tokens.shape[1])
                if passage_tokens is None:
                    passage_tokens = context_tokens
                else:
                    passage_tokens = torch.cat((passage_tokens, context_tokens), dim = 1)

            tokens_list.append(passage_tokens)
            tokens_len.append(passage_len)
        # 每个passage需要补长度
        max_len = max([sum(x) for x in tokens_len])
        for i,len_i in enumerate(tokens_len):
            len_i.insert(0, max_len - sum(len_i))
            padd_tokens = torch.full((1, len_i[0]), self.tokenizer.pad_token_id)
            tokens_list[i] = torch.cat((padd_tokens, tokens_list[i]), dim = 1)
        # 需要返回一个position_offsets tensor
        position_offsets = [offset[0] for offset in tokens_len]
        position_offsets = torch.tensor(position_offsets, dtype = torch.int).unsqueeze(0)
        ids = torch.cat(tokens_list, dim = 0).to(torch.long)
        # text = self.tokenizer.decode(ids[0])
        # print(text)
        if prompts_identical:
            position_offsets = None

        # Truncate prompt if generation would cause cache overflow
        tokens_start = [sum(tokens_len[0][:i]) for i in range(2,len(tokens_len[0]))]
        print(f'prompt length: {ids.shape[1]}\n{tokens_start}')
        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]
        else: overflow = 0
        
        # generate阶段传一个全可见的mask回去
        mask_2 = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        # 改mask
        mask_list = []
        sum_len = sum(tokens_len[0])
        for passage_index in range(batch_size):
            inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
            inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
            if passage_index == 0 and batch_size != 1: # pad + query
                current_start = tokens_len[passage_index][1]
                mask = torch.triu(torch.full((current_start, current_start), -65504.0),diagonal=1)
            elif mask_flag == False: # 如果不改mask,走input_masks
                mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
            else: # pad + sys + context + query
                system_len = tokens_len[passage_index][1]
                current_start = system_len
                question_len = tokens_len[passage_index][-1]
                system_context_len = sum(tokens_len[passage_index][1:-1])
                context_len = sum(tokens_len[passage_index][2:-1])
                sys_mask = torch.triu(torch.full((system_len, system_len), -65504.0),diagonal=1)
                context_sys_mask = torch.zeros(context_len, system_len)
                mask = torch.cat((sys_mask,context_sys_mask),dim=0)
                # 拼context
                for length in tokens_len[passage_index][2:-1]:
                    zero_mask = torch.full((current_start,length), -65504.0)
                    context_mask = torch.triu(torch.full((length, length), -65504.0),diagonal=1)
                    current_start += length
                    zero_mask_2 = torch.full((system_context_len - current_start,length), -65504.0)
                    tmp_mask = torch.cat((zero_mask,context_mask,zero_mask_2),dim=0)
                    mask = torch.cat((mask,tmp_mask),dim=1)
                    #尾部填充atten
                if tail_mask == True:
                    current_start = system_len
                    for length in tokens_len[passage_index][2:-1]:
                        current_start += length
                        mask[current_start-tail_len:current_start,system_len:current_start-length] = 0
                # 拼query
                zero_mask_3 = torch.full((system_context_len, question_len), -65504.0)
                mask = torch.cat((mask,zero_mask_3),dim=1)
                question_mask_1 = torch.zeros(question_len, system_context_len)
                question_mask_2 = torch.triu(torch.full((question_len, question_len), -65504.0),diagonal=1)
                question_mask = torch.cat((question_mask_1,question_mask_2),dim=1)
                mask = torch.cat((mask,question_mask),dim=0)

            mask = torch.cat((inf_mask_2,mask),dim=1)
            mask = torch.cat((inf_mask_1,mask),dim=0)
            mask = mask.unsqueeze(0)
            mask_list.append(mask)
        mask = torch.cat(mask_list, dim = 0)

        first_token = max(-overflow, 0)

        if last_layer_flag == True:
            whole_mask_list = []
            for passage_index in range(batch_size):
                inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
                inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
                whole_mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
                whole_mask = torch.cat((inf_mask_2,whole_mask),dim=1)
                whole_mask = torch.cat((inf_mask_1,whole_mask),dim=0)
                whole_mask = whole_mask.unsqueeze(0)
                whole_mask_list.append(whole_mask)
            whole_mask = torch.cat(whole_mask_list, dim = 0)
        
        else:
            whole_mask = None

        # Completion only

        if completion_only:
            first_token = ids.shape[-1]

        # Prepare for healing

        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]

        # Process prompt and begin gen
        # 如果传入的是additional_attn_mask
        q_len = ids.shape[1] - 1
        k_sub_all, v_sub_all, x_sub_all = self.cache_blend_warm_up(ids,
                             mask,
                             whole_mask,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings,
                             last_layer_flag=last_layer_flag,
                             last_layer=last_layer,rate=rate)
        k_sum = torch.sum(k_sub_all,dim=1)
        k_need_index = torch.topk(k_sum,int(q_len*rate)).indices
        k_need_index = k_need_index[k_sum[k_need_index] != 0]
        v_sum = torch.sum(v_sub_all,dim=1)
        v_need_index = torch.topk(v_sum,int(q_len*rate)).indices
        v_need_index = v_need_index[k_sum[v_need_index] != 0]
        k_need_index, indices = torch.sort(k_need_index)
        v_need_index, indices = torch.sort(v_need_index)
        x_sum = torch.sum(x_sub_all.squeeze(0),dim=1)
        x_need_index = torch.topk(x_sum,int(q_len*rate)).indices
        x_need_index = x_need_index[x_sum[x_need_index] != 0]
        x_need_index, indices = torch.sort(x_need_index)
        # 根据需要index重新设置mask
        # 注意，这里只改了单batch的情况
        # print(f'x_index: {x_need_index.tolist()}')
        print(f'x_index num: {len(x_need_index.tolist())}')
        for k_index in x_need_index.tolist():
            # if mask[0,k_index,tokens_len[0][1]:k_index] == 0:
            #     print(k_index)
            
            mask[0,k_index,tokens_len[0][1]:k_index] = 0
        if mask.dim() == 3:
            self.last_layer_nbce_gen_begin_base(ids,
                             mask,
                             whole_mask,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings,
                             last_layer_flag=last_layer_flag,
                             last_layer=last_layer)
        # 如果传入的是input_mask
        elif mask.dim() == 2:
            self._gen_begin_base(ids,
                             mask,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            if isinstance(prompt, str): return ""
            else: return [""] * len(prompt)

        # 这个修改对interlm20b来讲，会让模型不废话，效果是在generate之前会把prefill的所有文本都删除掉，对Qwen模型都不好用，会降低模型效果
        # self.sequence_ids = torch.zeros((batch_size, self.sequence_ids.shape[-1]), dtype = torch.int64)
        # Remove indexed embeddings from generator's sequence
        if input_embeddings is not None:
            self.sequence_ids[self.sequence_ids >= EMBEDDING_INDEX] = self.tokenizer.pad_token_id

        # Begin filters

        healed_token = []
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None
        # 这个函数内部是pass
        gen_settings.begin_filters(heal)

        # Generate tokens

        batch_eos = [False] * batch_size
        print('begin generate')
        for i in range(num_tokens):

            if self.abort_event and self.abort_event.is_set():
                break
            # 这里不能传additional_attn_mask，generate阶段要q_len对past_len全可见
            logits = self.model.forward(self.sequence_ids[:, -1:],
                                        self.cache,
                                        input_mask = mask_2,
                                        loras = loras,
                                        position_offsets = position_offsets,
                                        indexed_embeddings = input_embeddings).float().cpu()
            
            token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.nbce_sample(logits,
                                                                        gen_settings,
                                                                        self.sequence_ids,
                                                                        random.random(),
                                                                        self.tokenizer,
                                                                        prefix_token = unhealed_token
                                                                    )

            if unhealed_token is not None:
                unhealed_token_copy = unhealed_token
                healed_token = token

            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            # Post sampling hook

            if gen_settings.post_sampling_hooks:
                p = ExLlamaV2PostSamplingResult(
                    sampled_token = token,
                    sampled_prob = prob,
                    logits = logits,
                    candidate_tokens = None if ptokens.is_meta else ptokens,
                    candidate_probs = None if pprobs.is_meta else pprobs
                )
                for h in gen_settings.post_sampling_hooks:
                    h(p)
                token = p.sampled_token
                if p.feed_filters:
                    gen_settings.feed_filters(token)

            else:# 这个函数下面也是pass
                gen_settings.feed_filters(token)

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            unhealed_token = None
            if batch_size != 1:
                token = token[0]
            if eos or token.item() in self.stop_tokens: break

        # Decode

        decode_ids = self.sequence_ids[:, first_token:]
        if input_embeddings is not None:
            decode_ids = torch.stack([decode_ids[i][decode_ids[i] != self.tokenizer.pad_token_id] for i in range(batch_size)])

        if len(healed_token) and completion_only:
            decode_ids = torch.cat([healed_token, decode_ids], dim = -1)
        if self.tokenizer.decode(decode_ids[0,-1].unsqueeze(0), decode_special_tokens) == '[UNUSED_TOKEN_145]':
            text = self.tokenizer.decode(decode_ids[:,:-1], decode_special_tokens = decode_special_tokens)
        else:
            text = self.tokenizer.decode(decode_ids, decode_special_tokens = decode_special_tokens)
        if len(healed_token) and completion_only:
            pre_text = self.tokenizer.decode(unhealed_token_copy, decode_special_tokens = decode_special_tokens)
            text = [t[len(p):] for t, p in zip(text, pre_text)]

        if isinstance(prompt, str):
            return text[0]
        else:
            return text


    def cache_dump_generate_simple(self,
                        prompt: list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed: int or None = None,
                        token_healing: bool = False,
                        encode_special_tokens: bool = False,
                        decode_special_tokens: bool = False,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] | None = None,
                        stop_token: int or None = -1,
                        add_bos: bool = False,
                        abort_event: threading.Event | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        completion_only: bool = False,
                        mask_flag: bool = False,
                        tail_mask: bool = False,
                        tail_len: int = 50,
                        last_layer_flag: bool = False,
                        last_layer: int | None = None,
                        rate: float| None = None,
                        batch:int| None = None,
                        data_name:str | None = None,
                        model_name:str | None = None,
                        ):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """


        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed, inserting embeddings if provided

        batch_size = len(prompt)
        prompts_identical = batch_size == 1 or all(s == prompt[0] for s in prompt)


        # 由于要构造mask，所以要获取每段context的长度，所以得实现一个encode
        tokens_list = []
        tokens_len = []
        for passage in prompt:
            passage_tokens = None
            passage_len = []
            for context in passage:
                context_tokens = self.tokenizer.encode(context, add_bos = add_bos, 
                                                       encode_special_tokens = True)
                passage_len.append(context_tokens.shape[1])
                if passage_tokens is None:
                    passage_tokens = context_tokens
                else:
                    passage_tokens = torch.cat((passage_tokens, context_tokens), dim = 1)

            tokens_list.append(passage_tokens)
            tokens_len.append(passage_len)

        # # 需要返回一个position_offsets tensor
        system_token = tokens_list[0][:,:tokens_len[0][0]]
        system_len = system_token.shape[1]
        query_token = tokens_list[0][:,-tokens_len[0][-1]:]
        query_len = query_token.shape[1]
        all_context_tokens = []
        for index in range(len(tokens_len[0])):
            if index > 0 and index != len(tokens_len[0])-1:
                context_token = tokens_list[0][:,sum(tokens_len[0][:index]):sum(tokens_len[0][:index+1])]
                ids = torch.cat((system_token,context_token),dim=1).to(torch.long)
                all_context_tokens.append(ids)
        # text = self.tokenizer.decode(ids[0])
        # print(text)
        for (context_id,context_tokens) in enumerate(all_context_tokens):
                q_len = context_tokens.shape[1]
                mask = torch.triu(torch.full((q_len, q_len), -65504.0),diagonal=1).unsqueeze(0)
                position_offsets = torch.tensor(0, dtype = torch.int).unsqueeze(0)
                self._dump_gen_begin_base(context_tokens,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings,
                             context_id = context_id+1,index=batch,data_name=data_name,model_name=model_name)
                
    def cache_splice_generate_simple(self,
                        prompt: list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed: int or None = None,
                        token_healing: bool = False,
                        encode_special_tokens: bool = False,
                        decode_special_tokens: bool = False,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] | None = None,
                        stop_token: int or None = -1,
                        add_bos: bool = False,
                        abort_event: threading.Event | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        completion_only: bool = False,
                        mask_flag: bool = False,
                        tail_mask: bool = False,
                        tail_len: int = 50,
                        last_layer_flag: bool = False,
                        last_layer: int | None = None,
                        rate: float| None = None,
                        batch:int| None = None,
                        data_name:str| None = None,
                        model_name:str| None = None,
                        large_model:bool = False,
                        cacheBlend:bool = False):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """

        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed, inserting embeddings if provided

        batch_size = len(prompt)
        assert batch_size == 1
        prompts_identical = batch_size == 1 or all(s == prompt[0] for s in prompt)


        # 由于要构造mask，所以要获取每段context的长度，所以得实现一个encode
        tokens_list = []
        tokens_len = []
        for passage in prompt:
            passage_tokens = None
            passage_len = []
            for context in passage:
                context_tokens = self.tokenizer.encode(context, add_bos = add_bos, 
                                                       encode_special_tokens = True)
                passage_len.append(context_tokens.shape[1])
                if passage_tokens is None:
                    passage_tokens = context_tokens
                else:
                    passage_tokens = torch.cat((passage_tokens, context_tokens), dim = 1)

            tokens_list.append(passage_tokens)
            tokens_len.append(passage_len)

        if large_model:
            system_token = tokens_list[0][:,:tokens_len[0][0]]
            system_len = system_token.shape[1]
            query_token = tokens_list[0][:,-tokens_len[0][-1]:]
            query_len = query_token.shape[1]
            all_context_tokens = []
            for index in range(len(tokens_len[0])):
                if index > 0 and index != len(tokens_len[0])-1:
                    context_token = tokens_list[0][:,sum(tokens_len[0][:index]):sum(tokens_len[0][:index+1])]
                    ids = torch.cat((system_token,context_token),dim=1).to(torch.long)
                    all_context_tokens.append(ids)
            # text = self.tokenizer.decode(ids[0])
            # print(text)
            for (context_id,context_tokens) in enumerate(all_context_tokens):
                    q_len = context_tokens.shape[1]
                    position_offsets = torch.tensor(0, dtype = torch.int).unsqueeze(0)
                    self._dump_gen_begin_base(context_tokens,
                                 position_offsets = position_offsets,
                                 input_embeddings = input_embeddings,
                                 context_id = context_id+1,index=batch,data_name=data_name,model_name=model_name)
            print('KVCache Dump')
            

        # 每个passage需要补长度
        max_len = max([sum(x) for x in tokens_len])
        for i,len_i in enumerate(tokens_len):
            len_i.insert(0, max_len - sum(len_i))
            padd_tokens = torch.full((1, len_i[0]), self.tokenizer.pad_token_id)
            tokens_list[i] = torch.cat((padd_tokens, tokens_list[i]), dim = 1)
        
            
        # 需要返回一个position_offsets tensor
        position_offsets = [offset[0] for offset in tokens_len]
        position_offsets = torch.tensor(position_offsets, dtype = torch.int).unsqueeze(0)
        ids = torch.cat(tokens_list, dim = 0).to(torch.long)
        # text = self.tokenizer.decode(ids[0])
        # print(text)
        if prompts_identical:
            position_offsets = None

        # Truncate prompt if generation would cause cache overflow
        tokens_start = [sum(tokens_len[0][:i]) for i in range(2,len(tokens_len[0]))]
        print(f'prompt length: {ids.shape[1]}\n{tokens_start}')
        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]
        else: overflow = 0
        
        # generate阶段传一个全可见的mask回去
        mask_2 = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        # 改mask
        mask_list = []
        sum_len = sum(tokens_len[0])
        for passage_index in range(batch_size):
            inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
            inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
            if passage_index == 0 and batch_size != 1: # pad + query
                current_start = tokens_len[passage_index][1]
                mask = torch.triu(torch.full((current_start, current_start), -65504.0),diagonal=1)
            elif mask_flag == False: # 如果不改mask,走input_masks
                mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
            else: # pad + sys + context + query
                system_len = tokens_len[passage_index][1]
                current_start = system_len
                question_len = tokens_len[passage_index][-1]
                system_context_len = sum(tokens_len[passage_index][1:-1])
                context_len = sum(tokens_len[passage_index][2:-1])
                sys_mask = torch.triu(torch.full((system_len, system_len), -65504.0),diagonal=1)
                context_sys_mask = torch.zeros(context_len, system_len)
                mask = torch.cat((sys_mask,context_sys_mask),dim=0)
                # 拼context
                for length in tokens_len[passage_index][2:-1]:
                    zero_mask = torch.full((current_start,length), -65504.0)
                    context_mask = torch.triu(torch.full((length, length), -65504.0),diagonal=1)
                    current_start += length
                    zero_mask_2 = torch.full((system_context_len - current_start,length), -65504.0)
                    tmp_mask = torch.cat((zero_mask,context_mask,zero_mask_2),dim=0)
                    mask = torch.cat((mask,tmp_mask),dim=1)
                    #尾部填充atten
                if tail_mask == True:
                    current_start = system_len
                    for length in tokens_len[passage_index][2:-1]:
                        current_start += length
                        mask[current_start-tail_len:current_start,system_len:current_start-length] = 0
                # 拼query
                zero_mask_3 = torch.full((system_context_len, question_len), -65504.0)
                mask = torch.cat((mask,zero_mask_3),dim=1)
                question_mask_1 = torch.zeros(question_len, system_context_len)
                question_mask_2 = torch.triu(torch.full((question_len, question_len), -65504.0),diagonal=1)
                question_mask = torch.cat((question_mask_1,question_mask_2),dim=1)
                mask = torch.cat((mask,question_mask),dim=0)

            mask = torch.cat((inf_mask_2,mask),dim=1)
            mask = torch.cat((inf_mask_1,mask),dim=0)
            mask = mask.unsqueeze(0)
            mask_list.append(mask)
        mask = torch.cat(mask_list, dim = 0)

        first_token = max(-overflow, 0)

        if last_layer_flag == True:
            whole_mask_list = []
            for passage_index in range(batch_size):
                inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
                inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
                whole_mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
                whole_mask = torch.cat((inf_mask_2,whole_mask),dim=1)
                whole_mask = torch.cat((inf_mask_1,whole_mask),dim=0)
                whole_mask = whole_mask.unsqueeze(0)
                whole_mask_list.append(whole_mask)
            whole_mask = torch.cat(whole_mask_list, dim = 0)
        
        else:
            whole_mask = None

        # Completion only

        if completion_only:
            first_token = ids.shape[-1]

        # Prepare for healing

        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]
        ids = ids[:,:-tokens_len[0][-1]]
        q_len = ids.shape[1]
        tokens_len = [tokens_len[0][1:]]
        system_len = tokens_len[0][0]
        # 这里是根据计算选
        if rate != -1:

            k_change_mask, k_full_mask = self.cache_blend_select_index(ids,
                                 mask,
                                 whole_mask,
                                 loras,
                                 position_offsets = position_offsets,
                                 input_embeddings = input_embeddings,
                                 last_layer_flag=last_layer_flag,
                                 last_layer=last_layer,rate=rate,layer_index=1,tokens_len=tokens_len,
                                 data_name=data_name,model_name=model_name,batch = batch)
            k_sub_all = k_change_mask - k_full_mask
            k_sub_all = k_sub_all.squeeze(0)
            k_sub_all = k_sub_all.reshape(q_len,-1)
            k_sub_all = torch.abs(k_sub_all)
            k_sum = torch.sum(k_sub_all,dim=1)
            k_sum = k_sum.tolist()
            if not cacheBlend:
                k_sum = signal.detrend(k_sum, axis=0, type='linear', bp=0, overwrite_data=False)
            k_sum = k_sum[system_len:]
            k_sum = torch.tensor(k_sum,device=k_sub_all.device)
            k_need_index = torch.topk(k_sum,int(rate*(q_len - system_len))).indices.to('cpu')
            k_need_index = k_need_index + system_len
        else:
            # 测试直接返回正确答案的位置
            k_need_index = torch.arange(tokens_start[1],tokens_start[2])
        
        # x_need_index, indices = torch.sort(x_need_index)
        # print(k_need_index)

        if rate != 0: # rate 为0就什么都不要改了
            k_need_index = torch.cat((torch.arange(0,system_len),k_need_index),dim=0)
        
        query_len = tokens_len[0][-1]
        # 怎么把选择出来的k_need_index改变到ids上
        # ids = tokens_list[0][:,k_need_index]
        print(k_need_index.shape)
        # print(self.tokenizer.decode(ids[:,k_need_index.to('cpu')])[0])
        
        # self.cache.load_state(batch,tokens_len)
        ids  = tokens_list[0][:,:-query_len+1]
        # ids = ids[:,k_need_index]
        self.cache_retrival_gen_begin_base(input_ids=ids,
                                           k_need_index=k_need_index,
                                           tokens_len=tokens_len,
                                           position_offsets=position_offsets,
                                           input_embeddings=input_embeddings,
                                           batch=batch,
                                           data_name=data_name,
                                           model_name=model_name,
                                           )

        # self.cache.load_state(batch,tokens_len)
        ids  = tokens_list[0][:,-query_len+1:]
        self._gen_begin_base_with_cache_current_seq_len(ids,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings)
        if completion_only:
            first_token = tokens_list[0][:,:].shape[1]
        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]
        mask_2 = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        # self.sequence_ids = tokens_list[0][:,:-1]
        print("kvcache装载完成")
        batch_eos = [False] * batch_size
        healed_token = []
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None
        # 这个函数内部是pass
        gen_settings.begin_filters(heal)
        
        # Generate tokens
        print('begin generate')
        for i in range(num_tokens):
            if self.abort_event and self.abort_event.is_set():
                break
            # 这里不能传additional_attn_mask，generate阶段要q_len对past_len全可见
            logits = self.model.forward(self.sequence_ids[:, -1:],
                                        self.cache,
                                        input_mask = mask_2,
                                        loras = loras,
                                        position_offsets = position_offsets,
                                        indexed_embeddings = input_embeddings).float().cpu()
            
            token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.nbce_sample(logits,
                                                                        gen_settings,
                                                                        self.sequence_ids,
                                                                        random.random(),
                                                                        self.tokenizer,
                                                                        prefix_token = unhealed_token
                                                                    )

            if unhealed_token is not None:
                unhealed_token_copy = unhealed_token
                healed_token = token

            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            # Post sampling hook

            if gen_settings.post_sampling_hooks:
                p = ExLlamaV2PostSamplingResult(
                    sampled_token = token,
                    sampled_prob = prob,
                    logits = logits,
                    candidate_tokens = None if ptokens.is_meta else ptokens,
                    candidate_probs = None if pprobs.is_meta else pprobs
                )
                for h in gen_settings.post_sampling_hooks:
                    h(p)
                token = p.sampled_token
                if p.feed_filters:
                    gen_settings.feed_filters(token)

            else:# 这个函数下面也是pass
                gen_settings.feed_filters(token)

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            unhealed_token = None
            if batch_size != 1:
                token = token[0]
            if eos or token.item() in self.stop_tokens: break
        # if large_model:
        #     folder_path = f'./cacheDump/data/{data_name}/{model_name}'
        #     for file_name in os.listdir(folder_path):
        #         file_path = os.path.join(folder_path, file_name)
        #         try:
        #             os.unlink(file_path)
        #         except Exception as e:
        #             print(f'删除 {file_path} 失败. 原因: {e}')
        #     print('KVCache Clear')
        # Decode

        decode_ids = self.sequence_ids[:, first_token:]
        if input_embeddings is not None:
            decode_ids = torch.stack([decode_ids[i][decode_ids[i] != self.tokenizer.pad_token_id] for i in range(batch_size)])

        if len(healed_token) and completion_only:
            decode_ids = torch.cat([healed_token, decode_ids], dim = -1)
        if self.tokenizer.decode(decode_ids[0,-1].unsqueeze(0), decode_special_tokens) == '[UNUSED_TOKEN_145]':
            text = self.tokenizer.decode(decode_ids[:,:-1], decode_special_tokens = decode_special_tokens)
        else:
            text = self.tokenizer.decode(decode_ids, decode_special_tokens = decode_special_tokens)
        if len(healed_token) and completion_only:
            pre_text = self.tokenizer.decode(unhealed_token_copy, decode_special_tokens = decode_special_tokens)
            text = [t[len(p):] for t, p in zip(text, pre_text)]

        if isinstance(prompt, str):
            return text[0]
        else:
            return text
        

    def cache_splice_generate_simple_paper(self,
                        prompt: list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed: int or None = None,
                        token_healing: bool = False,
                        encode_special_tokens: bool = False,
                        decode_special_tokens: bool = False,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] | None = None,
                        stop_token: int or None = -1,
                        add_bos: bool = False,
                        abort_event: threading.Event | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        completion_only: bool = False,
                        mask_flag: bool = False,
                        tail_mask: bool = False,
                        tail_len: int = 50,
                        last_layer_flag: bool = False,
                        last_layer: int | None = None,
                        rate: float| None = None,
                        batch:int| None = None,
                        data_name:str| None = None,
                        model_name:str| None = None,
                        large_model:bool = False,
                        text_splitter:any | None = None,
                        retrival_model:any | None = None,
                        ):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """

        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed, inserting embeddings if provided

        batch_size = len(prompt)
        assert batch_size == 1
        prompts_identical = batch_size == 1 or all(s == prompt[0] for s in prompt)


        # 由于要构造mask，所以要获取每段context的长度，所以得实现一个encode
        tokens_list = []
        tokens_len = []
        for passage in prompt:
            passage_tokens = None
            passage_len = []
            for context in passage:
                context_tokens = self.tokenizer.encode(context, add_bos = add_bos, 
                                                       encode_special_tokens = True)
                passage_len.append(context_tokens.shape[1])
                if passage_tokens is None:
                    passage_tokens = context_tokens
                else:
                    passage_tokens = torch.cat((passage_tokens, context_tokens), dim = 1)

            tokens_list.append(passage_tokens)
            tokens_len.append(passage_len)

        if large_model:
            system_token = tokens_list[0][:,:tokens_len[0][0]]
            system_len = system_token.shape[1]
            query_token = tokens_list[0][:,-tokens_len[0][-1]:]
            query_len = query_token.shape[1]
            all_context_tokens = []
            for index in range(len(tokens_len[0])):
                if index > 0 and index != len(tokens_len[0])-1:
                    context_token = tokens_list[0][:,sum(tokens_len[0][:index]):sum(tokens_len[0][:index+1])]
                    ids = torch.cat((system_token,context_token),dim=1).to(torch.long)
                    all_context_tokens.append(ids)
            # text = self.tokenizer.decode(ids[0])
            # print(text)
            for (context_id,context_tokens) in enumerate(all_context_tokens):
                    q_len = context_tokens.shape[1]
                    position_offsets = torch.tensor(0, dtype = torch.int).unsqueeze(0)
                    self._dump_gen_begin_base(context_tokens,
                                 position_offsets = position_offsets,
                                 input_embeddings = input_embeddings,
                                 context_id = context_id+1,index=batch,data_name=data_name,model_name=model_name)
            print('KVCache Dump')
            

        # 每个passage需要补长度
        max_len = max([sum(x) for x in tokens_len])
        for i,len_i in enumerate(tokens_len):
            len_i.insert(0, max_len - sum(len_i))
            padd_tokens = torch.full((1, len_i[0]), self.tokenizer.pad_token_id)
            tokens_list[i] = torch.cat((padd_tokens, tokens_list[i]), dim = 1)
        
            
        # 需要返回一个position_offsets tensor
        position_offsets = [offset[0] for offset in tokens_len]
        position_offsets = torch.tensor(position_offsets, dtype = torch.int).unsqueeze(0)
        ids = torch.cat(tokens_list, dim = 0).to(torch.long)
        # text = self.tokenizer.decode(ids[0])
        # print(text)
        if prompts_identical:
            position_offsets = None

        # Truncate prompt if generation would cause cache overflow
        tokens_start = [sum(tokens_len[0][:i]) for i in range(2,len(tokens_len[0]))]
        print(f'prompt length: {ids.shape[1]}\n{tokens_start}')
        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]
        else: overflow = 0
        
        # generate阶段传一个全可见的mask回去
        mask_2 = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        # 改mask
        mask_list = []
        sum_len = sum(tokens_len[0])
        for passage_index in range(batch_size):
            inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
            inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
            if passage_index == 0 and batch_size != 1: # pad + query
                current_start = tokens_len[passage_index][1]
                mask = torch.triu(torch.full((current_start, current_start), -65504.0),diagonal=1)
            elif mask_flag == False: # 如果不改mask,走input_masks
                mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
            else: # pad + sys + context + query
                system_len = tokens_len[passage_index][1]
                current_start = system_len
                question_len = tokens_len[passage_index][-1]
                system_context_len = sum(tokens_len[passage_index][1:-1])
                context_len = sum(tokens_len[passage_index][2:-1])
                sys_mask = torch.triu(torch.full((system_len, system_len), -65504.0),diagonal=1)
                context_sys_mask = torch.zeros(context_len, system_len)
                mask = torch.cat((sys_mask,context_sys_mask),dim=0)
                # 拼context
                for length in tokens_len[passage_index][2:-1]:
                    zero_mask = torch.full((current_start,length), -65504.0)
                    context_mask = torch.triu(torch.full((length, length), -65504.0),diagonal=1)
                    current_start += length
                    zero_mask_2 = torch.full((system_context_len - current_start,length), -65504.0)
                    tmp_mask = torch.cat((zero_mask,context_mask,zero_mask_2),dim=0)
                    mask = torch.cat((mask,tmp_mask),dim=1)
                    #尾部填充atten
                if tail_mask == True:
                    current_start = system_len
                    for length in tokens_len[passage_index][2:-1]:
                        current_start += length
                        mask[current_start-tail_len:current_start,system_len:current_start-length] = 0
                # 拼query
                zero_mask_3 = torch.full((system_context_len, question_len), -65504.0)
                mask = torch.cat((mask,zero_mask_3),dim=1)
                question_mask_1 = torch.zeros(question_len, system_context_len)
                question_mask_2 = torch.triu(torch.full((question_len, question_len), -65504.0),diagonal=1)
                question_mask = torch.cat((question_mask_1,question_mask_2),dim=1)
                mask = torch.cat((mask,question_mask),dim=0)

            mask = torch.cat((inf_mask_2,mask),dim=1)
            mask = torch.cat((inf_mask_1,mask),dim=0)
            mask = mask.unsqueeze(0)
            mask_list.append(mask)
        mask = torch.cat(mask_list, dim = 0)

        first_token = max(-overflow, 0)

        if last_layer_flag == True:
            whole_mask_list = []
            for passage_index in range(batch_size):
                inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
                inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
                whole_mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
                whole_mask = torch.cat((inf_mask_2,whole_mask),dim=1)
                whole_mask = torch.cat((inf_mask_1,whole_mask),dim=0)
                whole_mask = whole_mask.unsqueeze(0)
                whole_mask_list.append(whole_mask)
            whole_mask = torch.cat(whole_mask_list, dim = 0)
        
        else:
            whole_mask = None

        # Completion only

        if completion_only:
            first_token = ids.shape[-1]

        # Prepare for healing

        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]
        ids = ids[:,:-tokens_len[0][-1]]
        q_len = ids.shape[1]
        tokens_len = [tokens_len[0][1:]]
        system_len = tokens_len[0][0]
        # 这里是根据计算选
        if rate != -1:
            k_need_index = []
            for i,start in enumerate(tokens_start[:-1]):
                end = tokens_start[i+1]
                k_need_index.extend(range(start,start+int((end-start)*rate)))
            k_need_index = torch.tensor(k_need_index,dtype=torch.int)
            # k_change_mask, k_full_mask = self.cache_blend_select_index(ids,
            #                      mask,
            #                      whole_mask,
            #                      loras,
            #                      position_offsets = position_offsets,
            #                      input_embeddings = input_embeddings,
            #                      last_layer_flag=last_layer_flag,
            #                      last_layer=last_layer,rate=rate,layer_index=1,tokens_len=tokens_len,
            #                      data_name=data_name,model_name=model_name,batch = batch)
            # k_sub_all = k_change_mask - k_full_mask
            # k_sub_all = k_sub_all.squeeze(0)
            # k_sub_all = k_sub_all.reshape(q_len,-1)
            # k_sub_all = torch.abs(k_sub_all)
            # k_sum = torch.sum(k_sub_all,dim=1)
            # k_sum = k_sum.tolist()
            # if not cacheBlend:
            #     k_sum = signal.detrend(k_sum, axis=0, type='linear', bp=0, overwrite_data=False)
            # k_sum = k_sum[system_len:]
            # k_sum = torch.tensor(k_sum,device=k_sub_all.device)
            # k_need_index = torch.topk(k_sum,int(rate*(q_len - system_len))).indices.to('cpu')
            # k_need_index = k_need_index + system_len
        else:
            # 测试直接返回正确答案的位置
            k_need_index = torch.arange(tokens_start[1],tokens_start[2])
        
        # x_need_index, indices = torch.sort(x_need_index)
        # print(k_need_index)

        if rate != 0: # rate 为0就什么都不要改了
            k_need_index = torch.cat((torch.arange(0,system_len),k_need_index),dim=0)
        
        query_len = tokens_len[0][-1]
        # 怎么把选择出来的k_need_index改变到ids上
        # ids = tokens_list[0][:,k_need_index]
        print(k_need_index.shape)
        # print(self.tokenizer.decode(ids[:,k_need_index.to('cpu')])[0])
        
        # self.cache.load_state(batch,tokens_len)
        ids  = tokens_list[0][:,:-query_len+1]
        # ids = ids[:,k_need_index]
        self.cache_retrival_gen_begin_base(input_ids=ids,
                                           k_need_index=k_need_index,
                                           tokens_len=tokens_len,
                                           position_offsets=position_offsets,
                                           input_embeddings=input_embeddings,
                                           batch=batch,
                                           data_name=data_name,
                                           model_name=model_name,
                                           )

        # self.cache.load_state(batch,tokens_len)
        ids  = tokens_list[0][:,-query_len+1:]
        self._gen_begin_base_with_cache_current_seq_len(ids,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings)
        if completion_only:
            first_token = tokens_list[0][:,:].shape[1]
        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]
        mask_2 = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        # self.sequence_ids = tokens_list[0][:,:-1]
        print("kvcache装载完成")
        batch_eos = [False] * batch_size
        healed_token = []
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None
        # 这个函数内部是pass
        gen_settings.begin_filters(heal)
        
        # Generate tokens
        print('begin generate')
        for i in range(num_tokens):
            if self.abort_event and self.abort_event.is_set():
                break
            # 这里不能传additional_attn_mask，generate阶段要q_len对past_len全可见
            logits = self.model.forward(self.sequence_ids[:, -1:],
                                        self.cache,
                                        input_mask = mask_2,
                                        loras = loras,
                                        position_offsets = position_offsets,
                                        indexed_embeddings = input_embeddings).float().cpu()
            
            token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.nbce_sample(logits,
                                                                        gen_settings,
                                                                        self.sequence_ids,
                                                                        random.random(),
                                                                        self.tokenizer,
                                                                        prefix_token = unhealed_token
                                                                    )

            if unhealed_token is not None:
                unhealed_token_copy = unhealed_token
                healed_token = token

            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            # Post sampling hook

            if gen_settings.post_sampling_hooks:
                p = ExLlamaV2PostSamplingResult(
                    sampled_token = token,
                    sampled_prob = prob,
                    logits = logits,
                    candidate_tokens = None if ptokens.is_meta else ptokens,
                    candidate_probs = None if pprobs.is_meta else pprobs
                )
                for h in gen_settings.post_sampling_hooks:
                    h(p)
                token = p.sampled_token
                if p.feed_filters:
                    gen_settings.feed_filters(token)

            else:# 这个函数下面也是pass
                gen_settings.feed_filters(token)

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            unhealed_token = None
            if batch_size != 1:
                token = token[0]
            if eos or token.item() in self.stop_tokens: break
        # if large_model:
        #     folder_path = f'./cacheDump/data/{data_name}/{model_name}'
        #     for file_name in os.listdir(folder_path):
        #         file_path = os.path.join(folder_path, file_name)
        #         try:
        #             os.unlink(file_path)
        #         except Exception as e:
        #             print(f'删除 {file_path} 失败. 原因: {e}')
        #     print('KVCache Clear')
        # Decode

        decode_ids = self.sequence_ids[:, first_token:]
        if input_embeddings is not None:
            decode_ids = torch.stack([decode_ids[i][decode_ids[i] != self.tokenizer.pad_token_id] for i in range(batch_size)])

        if len(healed_token) and completion_only:
            decode_ids = torch.cat([healed_token, decode_ids], dim = -1)
        if self.tokenizer.decode(decode_ids[0,-1].unsqueeze(0), decode_special_tokens) == '[UNUSED_TOKEN_145]':
            text = self.tokenizer.decode(decode_ids[:,:-1], decode_special_tokens = decode_special_tokens)
        else:
            text = self.tokenizer.decode(decode_ids, decode_special_tokens = decode_special_tokens)
        if len(healed_token) and completion_only:
            pre_text = self.tokenizer.decode(unhealed_token_copy, decode_special_tokens = decode_special_tokens)
            text = [t[len(p):] for t, p in zip(text, pre_text)]

        if isinstance(prompt, str):
            return text[0]
        else:
            return text
        
    def cache_splice_generate_simple_importance(self,
                        prompt: list,
                        gen_settings: ExLlamaV2Sampler.Settings,
                        num_tokens: int,
                        seed: int or None = None,
                        token_healing: bool = False,
                        encode_special_tokens: bool = False,
                        decode_special_tokens: bool = False,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] | None = None,
                        stop_token: int or None = -1,
                        add_bos: bool = False,
                        abort_event: threading.Event | None = None,
                        input_embeddings: torch.Tensor | None = None,
                        completion_only: bool = False,
                        mask_flag: bool = False,
                        tail_mask: bool = False,
                        tail_len: int = 50,
                        last_layer_flag: bool = False,
                        last_layer: int | None = None,
                        rate: float| None = None,
                        batch:int| None = None,
                        data_name:str| None = None,
                        model_name:str| None = None,
                        large_model:bool = False,
                        text_splitter:any | None = None,
                        retrival_model:any | None = None,
                        ):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """

        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed, inserting embeddings if provided

        batch_size = len(prompt)
        assert batch_size == 1
        prompts_identical = batch_size == 1 or all(s == prompt[0] for s in prompt)


        # 由于要构造mask，所以要获取每段context的长度，所以得实现一个encode
        tokens_list = []
        passage_token_list = []
        tokens_len = []
        for passage in prompt:
            passage_tokens = None
            passage_len = []
            for context in passage:
                context_tokens = self.tokenizer.encode(context, add_bos = add_bos, 
                                                       encode_special_tokens = True)
                passage_token_list.append(context_tokens)
                passage_len.append(context_tokens.shape[1])
                if passage_tokens is None:
                    passage_tokens = context_tokens
                else:
                    passage_tokens = torch.cat((passage_tokens, context_tokens), dim = 1)

            tokens_list.append(passage_tokens)
            tokens_len.append(passage_len)

        if large_model:
            query_prefix_len = self.tokenizer.encode(prompt[0][-1].split('Question: ')[0]).shape[1]
            system_token = tokens_list[0][:,:tokens_len[0][0]]
            system_len = system_token.shape[1]
            query_token = tokens_list[0][:,-tokens_len[0][-1]:]
            query_len = query_token.shape[1]
            all_context_tokens = []
            for index in range(len(tokens_len[0])):
                if index > 0:
                    context_token = tokens_list[0][:,sum(tokens_len[0][:index]):sum(tokens_len[0][:index+1])]
                    ids = torch.cat((system_token,context_token),dim=1).to(torch.long)
                    all_context_tokens.append(ids)
            # text = self.tokenizer.decode(ids[0])
            # print(text)
            for (context_id,context_tokens) in enumerate(all_context_tokens):
                    q_len = context_tokens.shape[1]
                    position_offsets = torch.tensor(0, dtype = torch.int).unsqueeze(0)
                    self._dump_gen_begin_base(context_tokens,
                                 position_offsets = position_offsets,
                                 input_embeddings = input_embeddings,
                                 context_id = context_id+1,
                                 index=batch,
                                 data_name=data_name,
                                 model_name=model_name,
                                 system_len=system_len,
                                 query_prefix_len=query_prefix_len,
                                 context_num = len(all_context_tokens))
            print('KVCache Dump')
            self.cache.importance.zero_()
            importance = self.cache.load_importance(batch, model_name, data_name, sum(tokens_len[0][1:-1]))[:,:sum(tokens_len[0][1:-1]),:]
            # system_token = tokens_list[0][:,:tokens_len[0][0]]
            # system_len = system_token.shape[1]
            # query_token = tokens_list[0][:,-tokens_len[0][-1]:]
            # query_len = query_token.shape[1]
            # all_context_tokens = []
            # for index in range(len(tokens_len[0])):
            #     if index > 0 and index != len(tokens_len[0])-1:
            #         context_token = tokens_list[0][:,sum(tokens_len[0][:index]):sum(tokens_len[0][:index+1])]
            #         ids = torch.cat((system_token,context_token),dim=1).to(torch.long)
            #         all_context_tokens.append(ids)
            # # text = self.tokenizer.decode(ids[0])
            # # print(text)
            # p_len = 0
            # for (context_id,context_tokens) in enumerate(all_context_tokens):
            #         q_len = context_tokens.shape[1]
            #         position_offsets = torch.tensor(0, dtype = torch.int).unsqueeze(0)
            #         self._dump_gen_begin_base_importance(context_tokens,
            #                      position_offsets = position_offsets,
            #                      input_embeddings = input_embeddings,
            #                      context_id = context_id+1,index=batch,data_name=data_name, model_name=model_name, p_len=p_len,system_len=system_len)
            #         p_len += (q_len - system_len)
            # importance = self.cache.importance[:,:sum(tokens_len[0][1:-1]),:].clone()
            # self.cache.dump_importance(batch, model_name, data_name, sum(tokens_len[0][1:-1]))
            # self.cache.importance.zero_()
            # print('KVCache Dump')
        else:
            self.cache.importance.zero_()
            importance = self.cache.load_importance(batch, model_name, data_name, sum(tokens_len[0][1:-1]))[:,:sum(tokens_len[0][1:-1]),:]
            

        # 每个passage需要补长度
        max_len = max([sum(x) for x in tokens_len])
        for i,len_i in enumerate(tokens_len):
            len_i.insert(0, max_len - sum(len_i))
            padd_tokens = torch.full((1, len_i[0]), self.tokenizer.pad_token_id)
            tokens_list[i] = torch.cat((padd_tokens, tokens_list[i]), dim = 1)
        
            
        # 需要返回一个position_offsets tensor
        position_offsets = [offset[0] for offset in tokens_len]
        position_offsets = torch.tensor(position_offsets, dtype = torch.int).unsqueeze(0)
        ids = torch.cat(tokens_list, dim = 0).to(torch.long)
        # text = self.tokenizer.decode(ids[0])
        # print(text)
        if prompts_identical:
            position_offsets = None

        # Truncate prompt if generation would cause cache overflow
        tokens_start = [sum(tokens_len[0][:i]) for i in range(2,len(tokens_len[0]))]
        print(f'prompt length: {ids.shape[1]}\n{tokens_start}')
        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]
        else: overflow = 0
        
        # generate阶段传一个全可见的mask回去
        mask_2 = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        # 改mask
        mask_list = []
        sum_len = sum(tokens_len[0])
        for passage_index in range(batch_size):
            inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
            inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
            if passage_index == 0 and batch_size != 1: # pad + query
                current_start = tokens_len[passage_index][1]
                mask = torch.triu(torch.full((current_start, current_start), -65504.0),diagonal=1)
            elif mask_flag == False: # 如果不改mask,走input_masks
                mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
            else: # pad + sys + context + query
                system_len = tokens_len[passage_index][1]
                current_start = system_len
                question_len = tokens_len[passage_index][-1]
                system_context_len = sum(tokens_len[passage_index][1:-1])
                context_len = sum(tokens_len[passage_index][2:-1])
                sys_mask = torch.triu(torch.full((system_len, system_len), -65504.0),diagonal=1)
                context_sys_mask = torch.zeros(context_len, system_len)
                mask = torch.cat((sys_mask,context_sys_mask),dim=0)
                # 拼context
                for length in tokens_len[passage_index][2:-1]:
                    zero_mask = torch.full((current_start,length), -65504.0)
                    context_mask = torch.triu(torch.full((length, length), -65504.0),diagonal=1)
                    current_start += length
                    zero_mask_2 = torch.full((system_context_len - current_start,length), -65504.0)
                    tmp_mask = torch.cat((zero_mask,context_mask,zero_mask_2),dim=0)
                    mask = torch.cat((mask,tmp_mask),dim=1)
                    #尾部填充atten
                if tail_mask == True:
                    current_start = system_len
                    for length in tokens_len[passage_index][2:-1]:
                        current_start += length
                        mask[current_start-tail_len:current_start,system_len:current_start-length] = 0
                # 拼query
                zero_mask_3 = torch.full((system_context_len, question_len), -65504.0)
                mask = torch.cat((mask,zero_mask_3),dim=1)
                question_mask_1 = torch.zeros(question_len, system_context_len)
                question_mask_2 = torch.triu(torch.full((question_len, question_len), -65504.0),diagonal=1)
                question_mask = torch.cat((question_mask_1,question_mask_2),dim=1)
                mask = torch.cat((mask,question_mask),dim=0)

            mask = torch.cat((inf_mask_2,mask),dim=1)
            mask = torch.cat((inf_mask_1,mask),dim=0)
            mask = mask.unsqueeze(0)
            mask_list.append(mask)
        mask = torch.cat(mask_list, dim = 0)

        first_token = max(-overflow, 0)

        if last_layer_flag == True:
            whole_mask_list = []
            for passage_index in range(batch_size):
                inf_mask_1 = torch.full((tokens_len[passage_index][0], sum_len), -65504)
                inf_mask_2 = torch.full((sum_len - tokens_len[passage_index][0], tokens_len[passage_index][0]), -65504)
                whole_mask = torch.triu(torch.full((sum_len - tokens_len[passage_index][0], sum_len -tokens_len[passage_index][0]), -65504.0),diagonal=1)
                whole_mask = torch.cat((inf_mask_2,whole_mask),dim=1)
                whole_mask = torch.cat((inf_mask_1,whole_mask),dim=0)
                whole_mask = whole_mask.unsqueeze(0)
                whole_mask_list.append(whole_mask)
            whole_mask = torch.cat(whole_mask_list, dim = 0)
        
        else:
            whole_mask = None

        # Completion only

        if completion_only:
            first_token = ids.shape[-1]

        # Prepare for healing

        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]
        ids = ids[:,:-tokens_len[0][-1]]
        q_len = ids.shape[1]
        tokens_len = [tokens_len[0][1:]]
        system_len = tokens_len[0][0]

        
        query_len = tokens_len[0][-1]


        # 让query去挑
        # 怎么把选择出来的k_need_index改变到ids上
        # ids = tokens_list[0][:,k_need_index]
        # print(self.tokenizer.decode(ids[:,k_need_index.to('cpu')])[0])

        # self.cache.load_state(batch,tokens_len)
        torch.set_printoptions(threshold=10_000)
        importance = torch.sum(importance[-1], dim=1)
        # importance = torch.sum(torch.sum(importance[-3:], dim=0), dim=1)
        chunk_score = []
        start = []
        end = []
        chunk_size = 1
        chunk_overlap = 1
        # 将token按64个token一个块划分，块与块之间重叠32个token
        tokens_start_without_prefix = torch.tensor(tokens_start,dtype=torch.int) - system_len
        tokens_start_without_prefix = tokens_start_without_prefix.tolist()
        for i,chunk_start in enumerate(tokens_start_without_prefix[:-1]):
            for mini_start in range(chunk_start,tokens_start_without_prefix[i+1],chunk_overlap):
                mini_end = min(tokens_start_without_prefix[i+1],mini_start+chunk_size)
                chunk_score.append(max(importance[mini_start:mini_end]))
                start.append(mini_start)
                end.append(mini_end)
        recompute_len = int(importance.shape[0] * rate)
        # 删除第一个context的块，因为重算它没有用
        idx = 0
        for s in start:
            if s < tokens_start_without_prefix[1]:
                idx+=1
        start = start[idx:]
        end = end[idx:]
        chunk_score = chunk_score[idx:]
        _, sort_chunk = torch.sort(torch.tensor(chunk_score),descending=True)
        recompute_index = []
        for i,chunk_index in enumerate(sort_chunk):
            if len(recompute_index) < recompute_len:
                recompute_index.extend(range(start[chunk_index],end[chunk_index]))
                recompute_index_set = set(recompute_index)
                recompute_index = list(recompute_index_set)
            else: break
        if len(recompute_index) > recompute_len:
            diff = len(recompute_index) - recompute_len
            _ , b = torch.sort(importance[start[sort_chunk[i-1]]:end[sort_chunk[i-1]]])
            b = b + start[sort_chunk[i-1]]
            b = set(b.tolist())
            recompute_index_set = set(recompute_index)
            joint = recompute_index_set & b
            joint_list = list(joint)
            _, bb = torch.sort(importance[joint_list])
            bb = bb + start[sort_chunk[i-1]]
            for j in bb[:diff].tolist():
                recompute_index.remove(j)
        k_need_index = torch.tensor(recompute_index,dtype=torch.int)
        k_need_index = k_need_index + system_len
        # importance = torch.sum(torch.sum(importance, dim=0), dim=1)
        # sorted_importance , k_need_index = torch.sort(importance,descending=True)
        # k_need_index += system_len
        # k_need_index = k_need_index[k_need_index>=tokens_start[1]]
        # ####
        # # k_need_index = torch.cat((torch.range(tokens_start[1],tokens_start[2])[:int(rate*(tokens_start[2]-tokens_start[1]))],k_need_index)).to(torch.int32)
        # k_need_index = k_need_index[:int(rate*sum(tokens_len[0][1:-1]))]
        if rate != 0:
            k_need_index = torch.cat((torch.range(0,system_len),k_need_index)).to(torch.int32)
        
        ids  = tokens_list[0][:,:-query_len+1]
        # ids = ids[:,k_need_index]
        self.cache_retrival_gen_begin_base(input_ids=ids,
                                           k_need_index=k_need_index,
                                           tokens_len=tokens_len,
                                           position_offsets=position_offsets,
                                           input_embeddings=input_embeddings,
                                           batch=batch,
                                           data_name=data_name,
                                           model_name=model_name,
                                           )

        # self.cache.load_state(batch,tokens_len)
        ids  = tokens_list[0][:,-query_len+1:]
        self._gen_begin_base_with_cache_current_seq_len(ids,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings)
        if completion_only:
            first_token = tokens_list[0][:,:].shape[1]
        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]
        mask_2 = self.tokenizer.padding_mask(ids) if batch_size > 1 else None
        # self.sequence_ids = tokens_list[0][:,:-1]
        print("kvcache装载完成")
        batch_eos = [False] * batch_size
        healed_token = []
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None
        # 这个函数内部是pass
        gen_settings.begin_filters(heal)
        
        # Generate tokens
        print('begin generate')
        for i in range(num_tokens):
            if self.abort_event and self.abort_event.is_set():
                break
            # 这里不能传additional_attn_mask，generate阶段要q_len对past_len全可见
            logits = self.model.forward(self.sequence_ids[:, -1:],
                                        self.cache,
                                        input_mask = mask_2,
                                        loras = loras,
                                        position_offsets = position_offsets,
                                        indexed_embeddings = input_embeddings).float().cpu()
            
            token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.nbce_sample(logits,
                                                                        gen_settings,
                                                                        self.sequence_ids,
                                                                        random.random(),
                                                                        self.tokenizer,
                                                                        prefix_token = unhealed_token
                                                                    )

            if unhealed_token is not None:
                unhealed_token_copy = unhealed_token
                healed_token = token

            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            # Post sampling hook

            if gen_settings.post_sampling_hooks:
                p = ExLlamaV2PostSamplingResult(
                    sampled_token = token,
                    sampled_prob = prob,
                    logits = logits,
                    candidate_tokens = None if ptokens.is_meta else ptokens,
                    candidate_probs = None if pprobs.is_meta else pprobs
                )
                for h in gen_settings.post_sampling_hooks:
                    h(p)
                token = p.sampled_token
                if p.feed_filters:
                    gen_settings.feed_filters(token)

            else:# 这个函数下面也是pass
                gen_settings.feed_filters(token)

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            unhealed_token = None
            if batch_size != 1:
                token = token[0]
            if eos or token.item() in self.stop_tokens: break
        # if large_model:
        #     folder_path = f'./cacheDump/data/{data_name}/{model_name}'
        #     for file_name in os.listdir(folder_path):
        #         file_path = os.path.join(folder_path, file_name)
        #         try:
        #             os.unlink(file_path)
        #         except Exception as e:
        #             print(f'删除 {file_path} 失败. 原因: {e}')
        #     print('KVCache Clear')
        # Decode

        decode_ids = self.sequence_ids[:, first_token:]
        if input_embeddings is not None:
            decode_ids = torch.stack([decode_ids[i][decode_ids[i] != self.tokenizer.pad_token_id] for i in range(batch_size)])

        if len(healed_token) and completion_only:
            decode_ids = torch.cat([healed_token, decode_ids], dim = -1)
        if self.tokenizer.decode(decode_ids[0,-1].unsqueeze(0), decode_special_tokens) == '[UNUSED_TOKEN_145]':
            text = self.tokenizer.decode(decode_ids[:,:-1], decode_special_tokens = decode_special_tokens)
        else:
            text = self.tokenizer.decode(decode_ids, decode_special_tokens = decode_special_tokens)
        if len(healed_token) and completion_only:
            pre_text = self.tokenizer.decode(unhealed_token_copy, decode_special_tokens = decode_special_tokens)
            text = [t[len(p):] for t, p in zip(text, pre_text)]

        if isinstance(prompt, str):
            return text[0]
        else:
            return text

def tokenize_file(tokenizer, text,idx, output_path='./retrival/'):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    token_ids = tokenizer.encode(text, add_bos = False, 
                                                   encode_special_tokens = True)[0]
    tokens = [tokenizer.decode(token_id.unsqueeze(0)) for token_id in token_ids]
                # encoded_input = tokenizer(text, return_tensors='pt', add_special_tokens=False)
                # token_ids = encoded_input['input_ids'][0]
                
    with open(output_path+f'{idx}.txt', 'w', encoding='utf-8') as file:
        for i, (token_id, token) in enumerate(zip(token_ids, tokens)):
            file.write(f"{i}, {token_id}, '{token}'\n")

def read_lookup_table(file_path):
    lookup_table = []
    # 定义正则表达式来匹配 idx, token_id, 和可能跨越多行的 token_string
    pattern = re.compile(r"(\d+),\s*(\d+),\s*'((?:[^']|'(?!$))*)'", re.MULTILINE)
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
        # 使用正则表达式匹配所有行
        matches = pattern.findall(content)
        for match in matches:
            idx, token_id, token_string = match
            # 替换代表换行符的单引号
            token_string = token_string.replace("\n'", "\n")
            lookup_table.append((int(idx), int(token_id), token_string))
    cat_text = ""
    position_idx_record = [] # 记录每个char在lookup_table中的idx
    for idx, _, token_string in lookup_table:
        cat_text += token_string
        position_idx_record+=[idx]*len(token_string)
    return lookup_table, cat_text, position_idx_record
def find_text_indices(text, cat_text, position_idx_record):
    # 初始化 start_idx 和 end_idx 为 None
    start_idx = end_idx = -1
    text = re.sub(r'\s', '',text)
    if start_idx == -1:
        target_regex = re.compile(".{0,5}".join(map(re.escape, text)), re.DOTALL)
        match = target_regex.search(cat_text)
        if match:
            start_idx = match.start()
    if end_idx == -1:
        target_regex = re.compile(".{0,5}".join(map(re.escape, text)), re.DOTALL)
        match = target_regex.search(cat_text)
        if match:
            end_idx = match.end()
    if end_idx <= start_idx:
        end_idx = start_idx = -1
    if start_idx != -1 and end_idx != -1:
        start_idx = position_idx_record[start_idx]
        end_idx = position_idx_record[end_idx]
        return start_idx, end_idx
    else:
        # raise Exception(f"Failed to find text: {text[:10]}")
        print(f"Failed to find text: {text[:10]}")
        return text, None
    
def compute_score(retrival_model, processed_chunk_list, query:str):        
    score_list = []
    query_output = retrival_model.encode(query, return_dense=True, return_sparse=True, return_colbert_vecs=True)
    for sentence in processed_chunk_list:
        weights_for_different_modes = [1, 1., 1.]
        weight_sum = 0
        weight_sum += 1
        weight_sum += 1
        weight_sum += 1
        all_score = 0
        dense_score = query_output['dense_vecs']@ sentence['dense_vecs'].T
        all_score += dense_score * weights_for_different_modes[0]
        lexical_weights_score = retrival_model.compute_lexical_matching_score(query_output['lexical_weights'], sentence['lexical_weights'])
        all_score += lexical_weights_score * weights_for_different_modes[1]
        colbert_score = retrival_model.colbert_score(query_output['colbert_vecs'], sentence['colbert_vecs'])
        all_score += colbert_score * weights_for_different_modes[2]
        
        all_score = all_score/weight_sum 
        score_list.append(all_score)
    return score_list
def get_top_k_idx(query_score_list, k=5):
    # 对每个query进行embedding的打分计算
    query_score_top_k_idx = []
    for query_score in query_score_list:
        query_score_top_k_idx.append(
            sorted(range(len(query_score)), key=lambda i: query_score[i], reverse=True)[:k]
        )
    return query_score_top_k_idx

def split_text_by_punctuation_and_overlap(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into chunks at punctuation, with specified size and overlap, without splitting words."""
    punctuation = ".!?,;:"  # Define punctuation marks used to split the text
    chunks = []
    position = 0
    text_length = len(text)

    while position < text_length:
        # Check if the remaining text is less than the chunk size
        if position + chunk_size >= text_length:
            chunks.append(text[position:])
            break

        # Find the end of the chunk ideally at a punctuation mark near the chunk size limit
        end = position + chunk_size
        while end < text_length and not (text[end] in punctuation and text[end + 1].isspace()):
            end += 1

        # Ensure the chunk ends at a space after punctuation
        if end < text_length:
            end += 1

        # Find the nearest word boundary after punctuation
        while end < text_length and not text[end].isspace():
            end += 1

        # Extract the chunk
        chunk = text[position:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Calculate the next start position with overlap, ensuring it starts at a word boundary
        position = max(end - chunk_overlap, position + 1)
        while position > 0 and not text[position - 1].isspace():
            position -= 1

    return chunks


