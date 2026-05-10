import functools
import itertools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.layers.sampler import get_token_ids_logprobs, get_top_logprobs
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sgl_jax.srt.managers.scheduler import GenerationBatchResult
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.speculative.eagle_util import (
    EagleDraftInput,
    EagleVerifyInput,
    EagleVerifyOutput,
    build_tree_kernel_efficient,
    build_tree_mask_for_draft_decode,
)
from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)
RETURN_ORIGINAL_LOGPROB = get_bool_env_var("RETURN_ORIGINAL_LOGPROB")


class EAGLEWorker(ModelWorker):
    def __init__(self, server_args, target_worker: ModelWorker):
        self.server_args = server_args
        self.target_worker = target_worker
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.req_to_token_pool, self.token_to_kv_pool_allocator = target_worker.get_memory_pool()
        self.hot_token_ids = None

        # Initialize dummy tensors for EAGLE operations
        self.num_new_pages_per_topk = None
        self.extend_lens = None

        # this must be put at last to make sure model state is correct
        super().__init__(
            server_args,
            target_worker.mesh,
            req_to_token_pool=self.req_to_token_pool,
            is_draft_worker=True,
        )
        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()

        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_model_runner.model, "load_lm_head_from_target")
                and self.draft_model_runner.model.load_lm_head_from_target
            ):
                self.draft_model_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_model_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_model_runner.model.hot_token_ids is not None:
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(NamedSharding(self.model_runner.mesh, P())),
                )
        else:
            if self.hot_token_ids is not None:
                head = head.clone()
                self.hot_token_ids = device_array(
                    self.draft_model_runner.model.hot_token_ids,
                    sharding=(NamedSharding(self.model_runner.mesh, P())),
                )
                head.data = head.data[self.hot_token_ids]

            # Share the embedding and lm_head
            self.draft_model_runner.model.set_embed_and_head(embed, head)

        self.model_runner.initialize_jit()
        (
            precompile_token_paddings,
            precompile_bs_paddings,
            precompile_cache_loc_paddings,
        ) = self.target_worker.get_precompile_paddings()
        self.precompile_bs_paddings = precompile_bs_paddings
        self.precompile_cache_loc_paddings = precompile_cache_loc_paddings
        self.precompile_token_paddings = precompile_token_paddings

    def forward_batch_speculative_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
    ):
        if model_worker_batch.forward_mode.is_extend():
            # FIXME(pc) add padding logic here
            # Only reshape temperatures if they're 1D (from_schedule_batch produces 1D,
            # but generate_for_precompile_all_greedy produces 2D with shape (bs, 1))

            if model_worker_batch.sampling_info.temperatures.ndim == 1:
                model_worker_batch.sampling_info.temperatures = (
                    model_worker_batch.sampling_info.temperatures[:, None]
                )
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                self.mesh,
                vocab_size=self.model_config.vocab_size,
            )
            # target extend
            logits_output, next_token_ids, cache_miss_count, bid, seq_lens = (
                self.forward_target_extend(model_worker_batch, sampling_metadata)
            )
            # draft extend for Update Draft State
            self.forward_draft_extend(
                model_worker_batch, logits_output.hidden_states, next_token_ids
            )
            # FIXME(pc) refactor this to batch output
            batch_output = GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                next_draft_input=model_worker_batch.spec_info,
                allocate_lens=model_worker_batch.seq_lens[: model_worker_batch.real_bs],
                bid=bid,
                cache_miss_count=cache_miss_count,
                extend_input_len_per_req=None,
                extend_logprob_start_len_per_req=None,
            )
            return batch_output

        else:
            import os

            _PROF = os.environ.get("EAGLE_PROFILE") == "1"
            cur_allocate_lens = model_worker_batch.spec_info.allocate_lens
            if _PROF:
                jax.block_until_ready(cur_allocate_lens)
                _t0 = time.perf_counter()
            self.draft(model_worker_batch)
            if _PROF:
                jax.block_until_ready(model_worker_batch.spec_info.draft_token)
                _t1 = time.perf_counter()
            batch_output = self.verify(model_worker_batch, cur_allocate_lens)
            if _PROF:
                jax.block_until_ready(batch_output.accept_lens)
                _t2 = time.perf_counter()
                if model_worker_batch.real_bs > 1:
                    rb = model_worker_batch.real_bs
                    nd = self.speculative_num_draft_tokens
                    logger.info(
                        "[EAGLE-DBG] bs=%d accept=%s predict=%s",
                        rb,
                        batch_output.accept_lens[:rb].tolist(),
                        np.asarray(batch_output.next_token_ids)[: rb * nd]
                        .reshape(rb, nd)
                        .tolist(),
                    )
            self.draft_extend_after_verify(model_worker_batch, batch_output)
            if _PROF:
                jax.block_until_ready(batch_output.next_draft_input.topk_p)
                _t3 = time.perf_counter()
                logger.info(
                    "[EAGLE-PROF] draft=%.1fms verify=%.1fms dext=%.1fms total=%.1fms",
                    (_t1 - _t0) * 1e3,
                    (_t2 - _t1) * 1e3,
                    (_t3 - _t2) * 1e3,
                    (_t3 - _t0) * 1e3,
                )
            return batch_output

    def forward_target_extend(
        self, model_worker_batch: ModelWorkerBatch, sample_meta_data: SamplingMetadata
    ) -> tuple[LogitsProcessorOutput, jax.Array, int, int, np.ndarray]:
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        logits_output, next_token_ids, cache_miss_count = (
            self.target_worker.forward_batch_generation(
                model_worker_batch, sampling_metadata=sample_meta_data
            )
        )
        return (
            logits_output,
            next_token_ids,
            cache_miss_count,
            model_worker_batch.bid,
            model_worker_batch.seq_lens,
        )

    def forward_draft_extend(
        self,
        model_worker_batch: ModelWorkerBatch,
        hidden_states: jax.Array,
        next_token_ids: jax.Array,
    ):
        # FIXME(pc) move this all prepare to prepare_for_extend_after_target_prefill
        verified_id_np = np.asarray(jax.device_get(next_token_ids))[: model_worker_batch.real_bs]
        model_worker_batch.spec_info = EagleDraftInput(
            hidden_states=hidden_states,
            verified_id=verified_id_np,
            num_tokens_per_batch=np.asarray(1, dtype=jnp.int32),
            num_tokens_for_logprob_per_batch=np.asarray(1, dtype=jnp.int32),
            allocate_lens=model_worker_batch.seq_lens,
        )
        model_worker_batch.return_hidden_states = False
        model_worker_batch.spec_info.prepare_for_extend_after_target_prefill(
            model_worker_batch=model_worker_batch
        )
        model_worker_batch.spec_info.capture_hidden_mode = CaptureHiddenMode.LAST
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.return_logprob = False

        # Set forward_metadata for draft_model_runner's attention backend
        forward_metadata = self.draft_model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )

        self.draft_model_runner.attn_backend.forward_metadata = forward_metadata
        forward_batch.forward_mode = ForwardMode.EXTEND
        # last_idx = np.cumsum(model_worker_batch.extend_seq_lens, axis=0) - 1

        logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        logits_output.next_token_logits = logits_output.next_token_logits[
            : model_worker_batch.real_bs, :
        ]
        if len(logits_output.hidden_states.shape) == 1:
            logits_output.hidden_states = jnp.expand_dims(logits_output.hidden_states, axis=0)
        assert isinstance(forward_batch.spec_info, EagleDraftInput)
        forward_batch.spec_info.allocate_lens = model_worker_batch.seq_lens[
            : model_worker_batch.real_bs
        ]

        self.capture_for_decode(logits_output, forward_batch.spec_info)

    def copy_model_worker_batch_to_cpu(self, model_worker_batch: ModelWorkerBatch):
        names = (
            "input_ids",
            "seq_lens",
            "out_cache_loc",
            "positions",
            "req_pool_indices",
            "cache_loc",
            "extend_prefix_lens",
            "extend_seq_lens",
        )
        vals = jax.device_get(tuple(getattr(model_worker_batch, n) for n in names))
        for n, v in zip(names, vals, strict=True):
            setattr(model_worker_batch, n, np.array(v, dtype=v.dtype) if v is not None else None)

    @property
    def draft_model_runner(self):
        return self.get_model_runner()

    def _reshard_logits(self, logits: jax.Array) -> jax.Array:
        # logits_processor outputs vocab-sharded P("data","tensor"); top_k can't
        # produce a (bs, k) result with k % tp != 0 under that spec. Fully
        # replicate so the downstream eagle host-side orchestration
        # (select_top_k_tokens / update_eagle_lists / build_tree) never hits
        # explicit-sharding scatter ambiguities.
        return jax.device_put(logits, NamedSharding(self.mesh, P()))

    def capture_for_decode(
        self, logits_output: LogitsProcessorOutput, draft_input: EagleDraftInput
    ):
        rep = NamedSharding(self.mesh, P())
        topk_p, topk_index = topk_probs_from_logits(
            logits_output.next_token_logits, self.topk
        )
        draft_input.topk_p = jax.device_put(topk_p, rep)
        draft_input.topk_index = jax.device_put(topk_index, rep)
        draft_input.hidden_states = jax.device_put(logits_output.hidden_states, rep)

    def get_padding_bs(self, real_bs: int) -> int:
        self.precompile_bs_paddings.sort()
        select_bs_index = -1
        bs_padding_size = 0
        for i, size in enumerate(self.precompile_bs_paddings):
            if size >= real_bs:
                bs_padding_size = size - real_bs
                select_bs_index = i
                break
        if select_bs_index < 0:
            raise RuntimeError("did not get comperate padding bs, it should not happened")
        return bs_padding_size, select_bs_index

    def padding_for_decode(self, model_worker_batch: ModelWorkerBatch):
        _, padding_bs_index = self.get_padding_bs(model_worker_batch.real_bs)
        self.copy_model_worker_batch_to_cpu(model_worker_batch)
        model_worker_batch.spec_info.prepare_for_draft_decode(
            model_worker_batch, self.topk, self.speculative_num_steps
        )
        # get unpadded seq_lens
        model_worker_batch.seq_lens = model_worker_batch.seq_lens
        seq_lens_cpu = model_worker_batch.seq_lens
        page_size = self.page_size
        token_indices_with_all_reqs = self.req_to_token_pool.req_to_token[
            model_worker_batch.req_pool_indices
        ]
        spec_info = model_worker_batch.spec_info
        assert isinstance(spec_info, EagleDraftInput)
        cache_loc_flat = np.array([], dtype=np.int32)
        if len(seq_lens_cpu) > 0:
            # Filter out empty sequences
            valid_mask = seq_lens_cpu > 0
            if np.any(valid_mask):
                valid_indices = np.where(valid_mask)[0]
                valid_allocate_lens = spec_info.allocate_lens[valid_mask]
                # Calculate aligned lengths for all valid sequences at once
                aligned_lengths = ((valid_allocate_lens + page_size - 1) // page_size) * page_size
                total_aligned_length = np.sum(aligned_lengths)
                # Pre-allocate the result array
                cache_loc_flat = np.zeros(total_aligned_length, dtype=np.int32)
                # Fill the array efficiently
                offset = 0
                for i, (seq_idx, allocate_len, aligned_len) in enumerate(
                    zip(valid_indices, valid_allocate_lens, aligned_lengths)
                ):
                    # Copy the actual data
                    cache_loc_flat[offset : offset + allocate_len] = token_indices_with_all_reqs[
                        seq_idx, :allocate_len
                    ]
                    # Padding is already zero from initialization
                    offset += aligned_len
        total_cache_loc_size = self.precompile_cache_loc_paddings[padding_bs_index]
        assert total_cache_loc_size >= len(cache_loc_flat)
        cache_loc_cpu = np.empty(total_cache_loc_size, dtype=np.int32)
        if len(cache_loc_flat) > 0:
            cache_loc_cpu[: len(cache_loc_flat)] = cache_loc_flat
        # Initialize padding area to ensure multiprocess consistency
        if len(cache_loc_flat) < total_cache_loc_size:
            cache_loc_cpu[len(cache_loc_flat) :] = 0

        model_worker_batch.cache_loc = cache_loc_cpu
        import os as _os

        if _os.environ.get("EAGLE_PROFILE") == "1" and len(cache_loc_flat) > 0:
            draft_pool_sz = self.draft_model_runner.max_total_num_tokens
            cmax = int(cache_loc_flat.max())
            if cmax >= draft_pool_sz:
                logger.error(
                    "[EAGLE-DBG] draft KV OOB: cache_loc.max=%d >= draft_pool=%d (target full slot)",
                    cmax,
                    draft_pool_sz,
                )
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST

        # out_cache_loc = model_worker_batch.out_cache_loc
        topk_index = spec_info.topk_index
        if self.hot_token_ids is not None:
            model_worker_batch.spec_info.topk_index = self.hot_token_ids[topk_index]
        # if we need custom mask, we should create for all at once and update it within loop
        # we should optimize build_tree_mask_for_draft_decode to a kernel
        if self.topk > 1:
            self.draft_model_runner.attn_backend.forward_metadata.custom_mask = (
                build_tree_mask_for_draft_decode(
                    model_worker_batch.seq_lens,
                    topk=topk_index.shape[1],
                    speculative_step_id=0,
                    parents_list=None,
                )
            )
        bs = self.precompile_bs_paddings[padding_bs_index]
        if bs - model_worker_batch.spec_info.verified_id.shape[0] > 0:
            model_worker_batch.spec_info.verified_id = np.pad(
                model_worker_batch.spec_info.verified_id,
                ((0, bs - model_worker_batch.spec_info.verified_id.shape[0]),),
            )
        if bs - model_worker_batch.spec_info.topk_p.shape[0] > 0:
            model_worker_batch.spec_info.topk_p = np.pad(
                model_worker_batch.spec_info.topk_p,
                (
                    (0, bs - model_worker_batch.spec_info.topk_p.shape[0]),
                    (0, 0),
                ),
            )
        if bs - model_worker_batch.seq_lens.shape[0] > 0:
            model_worker_batch.seq_lens = np.pad(
                model_worker_batch.seq_lens, ((0, bs - model_worker_batch.seq_lens.shape[0]),)
            )
            if model_worker_batch.spec_info.allocate_lens is not None:
                model_worker_batch.spec_info.allocate_lens = np.pad(
                    model_worker_batch.spec_info.allocate_lens,
                    ((0, bs - model_worker_batch.spec_info.allocate_lens.shape[0]),),
                )
        if bs - model_worker_batch.spec_info.topk_index.shape[0] > 0:
            model_worker_batch.spec_info.topk_index = np.pad(
                model_worker_batch.spec_info.topk_index,
                (
                    (0, bs - model_worker_batch.spec_info.topk_index.shape[0]),
                    (0, 0),
                ),
            )
        if bs - model_worker_batch.spec_info.hidden_states.shape[0] > 0:
            model_worker_batch.spec_info.hidden_states = np.pad(
                model_worker_batch.spec_info.hidden_states,
                (
                    (0, bs - model_worker_batch.spec_info.hidden_states.shape[0]),
                    (0, 0),
                ),
            )
        # Forward multiple steps
        model_worker_batch.speculative_eagle_topk = self.topk
        model_worker_batch.speculative_num_steps = self.speculative_num_steps
        model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
        model_worker_batch.input_ids = np.empty(bs * self.topk, np.int32)
        model_worker_batch.positions = np.empty(bs * self.topk, np.int32)

    def draft(self, model_worker_batch: ModelWorkerBatch):
        import os as _os

        _PROF = _os.environ.get("EAGLE_PROFILE") == "1"
        if _PROF:
            _d0 = time.perf_counter()
        self.padding_for_decode(model_worker_batch)
        if _PROF:
            _d1 = time.perf_counter()
        score_list, token_list, parents_list = self.draft_forward(model_worker_batch)
        if _PROF:
            jax.block_until_ready(token_list)
            _d2 = time.perf_counter()
        verified_seq_lens = model_worker_batch.seq_lens - 1
        max_seq_len = int(np.max(verified_seq_lens)) if verified_seq_lens.size > 0 else 1
        max_context_len = self._pick_context_len(max_seq_len)
        (
            tree_mask,
            position,
            retrive_index,
            retrive_next_token,
            retrive_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            model_worker_batch.spec_info.verified_id,
            score_list,
            token_list,
            parents_list,
            verified_seq_lens,
            np.sum(verified_seq_lens),
            self.topk,
            self.speculative_num_draft_tokens,
            max_context_len,
            model_worker_batch.seq_lens.shape[0],
            model_worker_batch.speculative_num_steps,
            self.mesh,
        )
        if _PROF:
            jax.block_until_ready(draft_tokens)
            _d3 = time.perf_counter()
            logger.info(
                "[EAGLE-DPROF] pad=%.1f loop=%.1f tree=%.1f",
                (_d1 - _d0) * 1e3,
                (_d2 - _d1) * 1e3,
                (_d3 - _d2) * 1e3,
            )
        model_worker_batch.spec_info = EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrive_index=retrive_index,
            retrive_next_token=retrive_next_token,
            retrive_next_sibling=retrive_next_sibling,
            retrive_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            seq_lens_sum=model_worker_batch.seq_lens_sum,
            seq_lens_cpu=model_worker_batch.seq_lens,
        )
        return model_worker_batch.spec_info

    def _pick_context_len(self, max_seq_len: int) -> int:
        max_seq_len = max(int(max_seq_len), 1)
        if self.precompile_token_paddings:
            for padding in self.precompile_token_paddings:
                if padding >= max_seq_len:
                    return padding
        return 1 << (max_seq_len - 1).bit_length()

    def verify(self, model_worker_batch: ModelWorkerBatch, cur_allocate_lens: jax.Array):
        import os as _os

        _PROF = _os.environ.get("EAGLE_PROFILE") == "1"
        spec_info: EagleVerifyInput = model_worker_batch.spec_info
        spec_info.allocate_lens = cur_allocate_lens
        if _PROF:
            _v0 = time.perf_counter()
        spec_info.prepare_for_verify(model_worker_batch, self.page_size, self.target_worker)
        forward_metadata = self.target_worker.model_runner.attn_backend.get_eagle_forward_metadata(
            model_worker_batch
        )
        if _PROF:
            jax.block_until_ready(forward_metadata.custom_mask)
            _v1 = time.perf_counter()

        logits_output, _, cache_miss_count = self.target_worker.forward_batch_generation(
            model_worker_batch, skip_sample=True, forward_metadata=forward_metadata
        )
        rep = NamedSharding(self.mesh, P())
        logits_output.next_token_logits = jax.device_put(logits_output.next_token_logits, rep)
        logits_output.hidden_states = jax.device_put(logits_output.hidden_states, rep)
        if _PROF:
            jax.block_until_ready(logits_output.next_token_logits)
            _v2 = time.perf_counter()
        spec_info.hidden_states = logits_output.hidden_states
        (
            predict,
            verified_id,
            accept_length,
            accept_index,
        ) = spec_info.sample(
            model_worker_batch,
            logits_output,
            self.model_runner.rngs,
            self.mesh,
        )
        # accept_index uses -1 for rejected slots; gathering positions/hidden with
        # -1 picks the *global* last element (= last req's last draft pos). dext
        # then writes those rejected tokens' draft-KV at that foreign position
        # inside *each* req's own page, corrupting prefix KV for all but the last
        # req. Redirect -1 to each req's own last draft index so the junk write
        # lands on a slot that req will never read (beyond its accept length).
        draft_n = self.speculative_num_draft_tokens
        per_req_last = (np.arange(len(accept_index)) // draft_n) * draft_n + draft_n - 1
        safe_index = np.where(accept_index >= 0, accept_index, per_req_last)
        logits_output.next_token_logits = logits_output.next_token_logits[safe_index, :]
        logits_output.hidden_states = logits_output.hidden_states[safe_index, :]
        model_worker_batch.positions = model_worker_batch.positions[safe_index]
        new_seq_lens = model_worker_batch.seq_lens + accept_length
        next_draft_input = EagleDraftInput(
            verified_id=verified_id,
            new_seq_lens=new_seq_lens,
            allocate_lens=cur_allocate_lens,
            hidden_states=logits_output.hidden_states,
        )

        if _PROF:
            _v3 = time.perf_counter()
            logger.info(
                "[EAGLE-VPROF] meta=%.1f fwd=%.1f sample=%.1f",
                (_v1 - _v0) * 1e3,
                (_v2 - _v1) * 1e3,
                (_v3 - _v2) * 1e3,
            )
        model_worker_batch.spec_info = next_draft_input
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            next_draft_input=next_draft_input,
            accept_lens=accept_length,
            # FIXME(pc) this field is for overlap
            allocate_lens=cur_allocate_lens,
            bid=model_worker_batch.bid,
            cache_miss_count=cache_miss_count,
            extend_input_len_per_req=None,
            extend_logprob_start_len_per_req=None,
        )

    def add_logprob_values(
        self,
        batch: ScheduleBatch,
        res: EagleVerifyOutput,
        logits_output: LogitsProcessorOutput,
    ):
        # Extract args
        logits_output = res.logits_output
        top_logprobs_nums = batch.top_logprobs_nums
        token_ids_logprobs = batch.token_ids_logprobs
        accepted_indices = res.accepted_indices
        assert len(accepted_indices) == len(logits_output.next_token_logits)

        temperatures = batch.sampling_info.temperatures
        num_draft_tokens = batch.spec_info.draft_token_num
        # acceptance indices are the indices in a "flattened" batch.
        # dividing it to num_draft_tokens will yield the actual batch index.
        temperatures = temperatures[accepted_indices // num_draft_tokens]
        if RETURN_ORIGINAL_LOGPROB:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits, axis=-1)
        else:
            logprobs = jax.nn.log_softmax(logits_output.next_token_logits / temperatures, axis=-1)
        batch_next_token_ids = res.verified_id
        num_tokens_per_req = [accept + 1 for accept in res.accept_length_per_req_cpu]

        # We should repeat top_logprobs_nums to match num_tokens_per_req.
        top_logprobs_nums_repeat_interleaved = []
        token_ids_logprobs_repeat_interleaved = []
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req):
            top_logprobs_nums_repeat_interleaved.extend([num] * num_tokens)
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req):
            token_ids_logprobs_repeat_interleaved.extend([token_ids] * num_tokens)

        # Extract logprobs.
        # NOTE: get_top_logprobs / get_token_ids_logprobs now return device
        # dense tensors; per-req trimming happens on host downstream. This
        # spec-decode path doesn't currently route through that host
        # slicing, so it is best-effort and may need follow-up alongside
        # the broader spec+DP+logprob unification.
        if any(x > 0 for x in top_logprobs_nums):
            (
                logits_output.next_token_top_logprobs_val,
                logits_output.next_token_top_logprobs_idx,
            ) = get_top_logprobs(
                logprobs,
                top_logprobs_nums_repeat_interleaved,
            )

        if any(x is not None for x in token_ids_logprobs):
            logits_output.next_token_token_ids_logprobs_val = get_token_ids_logprobs(
                logprobs,
                token_ids_logprobs_repeat_interleaved,
                None,
            )
            logits_output.next_token_token_ids_logprobs_idx = None

        logits_output.next_token_logprobs = logprobs[
            jnp.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
            batch_next_token_ids,
        ]

        # Add output logprobs to the request
        pt = 0
        next_token_logprobs = logits_output.next_token_logprobs.tolist()
        verified_ids = batch_next_token_ids.tolist()
        for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
            for _ in range(num_tokens):
                if req.return_logprob:
                    req.output_token_logprobs_val.append(next_token_logprobs[pt])
                    req.output_token_logprobs_idx.append(verified_ids[pt])
                    if req.top_logprobs_num > 0:
                        req.output_top_logprobs_val.append(
                            res.logits_output.next_token_top_logprobs_val[pt]
                        )
                        req.output_top_logprobs_idx.append(
                            res.logits_output.next_token_top_logprobs_idx[pt]
                        )
                pt += 1

    def draft_extend_after_verify(
        self, model_worker_batch: ModelWorkerBatch, batch_output: GenerationBatchResult
    ):
        if batch_output.next_draft_input.verified_id.shape[0] <= 0:
            return
        draft_input = EagleDraftInput(
            hidden_states=batch_output.logits_output.hidden_states,
            allocate_lens=batch_output.allocate_lens,
        )
        model_worker_batch, logits_meatadata = draft_input.prepare_for_extend_after_verify(
            model_worker_batch,
            self.draft_model_runner,
            batch_output,
            self.speculative_num_draft_tokens,
        )

        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        if forward_batch.input_ids.shape[0] <= 0:
            return
        import os as _os

        if _os.environ.get("EAGLE_PROFILE") == "1" and model_worker_batch.real_bs > 1:
            md = self.draft_model_runner.attn_backend.forward_metadata
            logger.info(
                "[DEXT-DBG] bs=%d seq_lens=%s page_idx[:8]=%s cu_kv=%s cu_q=%s alloc=%s",
                model_worker_batch.real_bs,
                np.asarray(model_worker_batch.seq_lens).tolist(),
                np.asarray(md.page_indices)[:8].tolist(),
                np.asarray(md.cu_kv_lens).tolist(),
                np.asarray(md.cu_q_lens).tolist(),
                np.asarray(model_worker_batch.spec_info.allocate_lens).tolist(),
            )
        draft_logits_output, _, _ = self.draft_model_runner.forward(
            forward_batch,
            logits_metadata=logits_meatadata,
        )
        select_index = (
            np.arange(len(model_worker_batch.seq_lens[: model_worker_batch.real_bs]))
            * (self.speculative_num_steps + 1)
            + batch_output.accept_lens[: model_worker_batch.real_bs]
            - 1
        )
        topk_p, topk_index, hidden_sel = _dext_post_forward(
            draft_logits_output.next_token_logits,
            draft_logits_output.hidden_states,
            device_array(select_index, sharding=NamedSharding(self.mesh, P())),
            self.topk,
        )

        # prepare for next draft decode
        batch_output.next_draft_input.hidden_states = hidden_sel
        batch_output.next_draft_input.topk_p = topk_p
        batch_output.next_draft_input.topk_index = topk_index
        batch_output.next_draft_input.verified_id = batch_output.next_draft_input.verified_id[
            select_index
        ]
        batch_output.allocate_lens = batch_output.allocate_lens[: model_worker_batch.real_bs]
        batch_output.accept_lens = batch_output.accept_lens[: model_worker_batch.real_bs]

    def draft_forward(self, model_worker_batch: ModelWorkerBatch):
        topk_p, topk_index, hidden_states = (
            model_worker_batch.spec_info.topk_p,
            model_worker_batch.spec_info.topk_index,
            model_worker_batch.spec_info.hidden_states,
        )
        bs = model_worker_batch.seq_lens.shape[0]
        step_min_1 = self.speculative_num_steps - 1
        score_list: jax.Array = jnp.empty((bs, 1 + step_min_1 * self.topk, self.topk))
        token_list: jax.Array = jnp.empty(
            (bs, self.topk + step_min_1 * self.topk * self.topk), dtype=jnp.int32
        )
        parents_list: jax.Array = jnp.empty((bs, self.topk + 1 + step_min_1 * self.topk))
        scores = None
        positions_base = device_array(
            np.repeat(model_worker_batch.seq_lens, self.topk),
            sharding=(NamedSharding(self.model_runner.mesh, P())),
        )
        logits_metadata = None
        metadata_per_step = self.draft_model_runner.attn_backend.get_eagle_multi_step_metadata(
            model_worker_batch,
        )
        assert isinstance(metadata_per_step, list)
        # we just use logits_metadata's forward mode and capture mode, it will not be modified within the loop
        logits_metadata = LogitsMetadata.from_model_worker_batch(
            model_worker_batch, self.draft_model_runner.mesh
        )
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.draft_model_runner)
        forward_batch.out_cache_loc = np.empty((1,))
        forward_batch.cache_loc = np.empty((1,))
        forward_batch.spec_info = EagleDraftInput()
        forward_batch.spec_info.hidden_states = jnp.empty((bs * self.topk, hidden_states.shape[1]))
        for i in range(self.speculative_num_steps):

            (
                input_ids,
                hidden_states,
                scores,
                positions,
                score_list,
                token_list,
                parents_list,
            ) = _draft_step_pre(
                i,
                topk_p,
                topk_index,
                hidden_states,
                scores,
                score_list,
                token_list,
                parents_list,
                positions_base,
                self.topk,
            )
            if i == self.speculative_num_steps - 1:
                break

            forward_batch.input_ids = input_ids
            forward_batch.spec_info.hidden_states = hidden_states
            forward_batch.positions = positions
            self.draft_model_runner.attn_backend.forward_metadata = metadata_per_step[i]

            # Run forward
            forward_batch.bid = model_worker_batch.bid
            logits_output, _, _ = self.draft_model_runner.forward(
                forward_batch,
                logits_metadata=logits_metadata,
            )

            topk_p, topk_index, hidden_states = _draft_post_forward(
                logits_output.next_token_logits, logits_output.hidden_states, self.topk
            )

            if self.hot_token_ids is not None:
                topk_index = self.hot_token_ids[topk_index]

        return score_list, token_list, parents_list

    def run_spec_decode_precompile(self):
        self.precompile_spec_extend()
        self.precompile_spec_decode()
        # FIXME precompile some kernel

    def precompile_spec_extend(self):
        start_time = time.perf_counter()
        logger.info(
            "[SPEC_EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s",
            self.precompile_bs_paddings[-1:],
            self.precompile_token_paddings,
        )

        bs, _ = self.get_max_padded_size()
        pairs = list(itertools.product([bs], self.precompile_token_paddings))

        with tqdm(pairs, desc="[SPEC_EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                pair = list(pair)
                bs, num_tokens = pair[0], pair[1]
                pbar.set_postfix(bs=bs, tokens=num_tokens)
                if bs > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs, num_tokens)
                    continue
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.precompile_cache_loc_paddings[-1],
                    do_penalties=False,
                    speculative_algotithm=self.speculative_algorithm,
                )
                self.forward_batch_speculative_generation(model_worker_batch)
        end_time = time.perf_counter()
        logger.info("[SPEC_EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_spec_decode(self):
        start_time = time.perf_counter()
        logger.info(
            "[SPEC_DECODE] Begin to precompile bs_paddings=%s",
            self.precompile_bs_paddings,
        )

        with tqdm(
            self.precompile_bs_paddings, desc="[SPEC_DECODE] PRECOMPILE", leave=False
        ) as pbar:
            for bs in pbar:
                pbar.set_postfix(bs=bs)
                # use same page aligned with precompile cache_loc_paddings
                aligned_cache_loc_size = (
                    (bs * self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
                )
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    bs,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    do_penalties=False,
                    speculative_algotithm=self.speculative_algorithm,
                )
                spec_info = EagleDraftInput(
                    # FIXME(pc) dtype should according to serverargs
                    topk_p=jnp.ones(
                        (bs, self.topk),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    topk_index=jnp.ones((bs, self.topk), dtype=jnp.int32),
                    hidden_states=jnp.ones(
                        (bs, self.model_config.hidden_size),
                        dtype=jnp.bfloat16 if self.server_args.dtype == "bfloat16" else jnp.float32,
                    ),
                    verified_id=jnp.ones((bs,), dtype=jnp.int32),
                    accept_length=jnp.ones((bs,), dtype=jnp.int32),
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                    allocate_lens=model_worker_batch.seq_lens
                    + EagleDraftInput.ALLOC_LEN_PER_DECODE,
                )
                model_worker_batch.capture_hidden_mode = CaptureHiddenMode.LAST
                model_worker_batch.spec_info = spec_info
                model_worker_batch.speculative_eagle_topk = self.topk
                model_worker_batch.speculative_num_draft_tokens = self.speculative_num_draft_tokens
                model_worker_batch.speculative_num_steps = self.speculative_num_steps
                self.forward_batch_speculative_generation(model_worker_batch)

        end_time = time.perf_counter()
        logger.info("[SPEC_DECODE] Precompile finished in %.0f secs", end_time - start_time)


@jax.jit
def _verify_post_gather(logits, hidden, positions, accept_index):
    return logits[accept_index, :], hidden[accept_index, :], positions[accept_index]


@functools.partial(jax.jit, static_argnames=["i", "topk"])
def _draft_step_pre(
    i: int,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores,
    score_list: jax.Array,
    token_list: jax.Array,
    parents_list: jax.Array,
    positions_base: jax.Array,
    topk: int,
):
    """select_top_k + update_eagle_lists + positions in one dispatch."""
    if i == 0:
        input_ids, hidden_states, scores, tree_info = select_top_k_tokens_step_0(
            topk_p, topk_index, hidden_states, scores, topk
        )
    else:
        input_ids, hidden_states, scores, tree_info = select_top_k_tokens_step_greater_0(
            jnp.asarray(i), topk_p, topk_index, hidden_states, scores, topk
        )
    score_list, token_list, parents_list = update_eagle_lists(
        i, score_list, token_list, parents_list, tree_info, topk
    )
    return input_ids, hidden_states, scores, positions_base + i, score_list, token_list, parents_list


@functools.partial(jax.jit, static_argnames=["topk"])
def _dext_post_forward(
    next_token_logits: jax.Array, hidden_states: jax.Array, select_index: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fuse reshard + per-req gather + top_k after draft_extend into one dispatch."""
    sh = jax.typeof(next_token_logits).sharding
    if isinstance(sh, NamedSharding):
        rep = NamedSharding(sh.mesh, P())
        next_token_logits = jax.sharding.reshard(next_token_logits, rep)
        hidden_states = jax.sharding.reshard(hidden_states, rep)
    next_token_logits = next_token_logits[select_index]
    hidden_states = hidden_states[select_index]
    topk_logits, topk_index = jax.lax.top_k(next_token_logits, topk)
    lse = jax.nn.logsumexp(next_token_logits, axis=-1, keepdims=True)
    return jnp.exp(topk_logits - lse), topk_index, hidden_states


@functools.partial(jax.jit, static_argnames=["topk"])
def _draft_post_forward(
    next_token_logits: jax.Array, hidden_states: jax.Array, topk: int
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fuse the per-step reshard + top_k after draft forward into one dispatch."""
    sh = jax.typeof(next_token_logits).sharding
    if isinstance(sh, NamedSharding):
        rep = NamedSharding(sh.mesh, P())
        next_token_logits = jax.sharding.reshard(next_token_logits, rep)
        hidden_states = jax.sharding.reshard(hidden_states, rep)
    topk_logits, topk_index = jax.lax.top_k(next_token_logits, topk)
    lse = jax.nn.logsumexp(next_token_logits, axis=-1, keepdims=True)
    return jnp.exp(topk_logits - lse), topk_index, hidden_states


@functools.partial(jax.jit, static_argnames=["topk"])
def topk_probs_from_logits(
    logits: jax.Array, topk: int, axis: int = -1
) -> tuple[jax.Array, jax.Array]:
    """Return top-k probabilities without materializing the full softmax tensor."""
    working_logits = jnp.moveaxis(logits, axis, -1) if axis != -1 else logits
    # logits arrive vocab-sharded; replicate inside jit so callers don't need
    # a separate host-side device_put round-trip per step.
    sh = jax.typeof(working_logits).sharding
    if isinstance(sh, NamedSharding):
        working_logits = jax.sharding.reshard(working_logits, NamedSharding(sh.mesh, P()))
    topk_logits, topk_index = jax.lax.top_k(working_logits, topk)
    logsumexp = jax.nn.logsumexp(working_logits, axis=-1, keepdims=True)
    topk_probs = jnp.exp(topk_logits - logsumexp)

    if axis != -1:
        topk_probs = jnp.moveaxis(topk_probs, -1, axis)
        topk_index = jnp.moveaxis(topk_index, -1, axis)

    return topk_probs, topk_index


def fast_topk(values, topk, axis=-1):
    working_values = jnp.moveaxis(values, axis, -1) if axis != -1 else values
    result_vals, result_indices = jax.lax.top_k(working_values, topk)

    if axis != -1:
        result_vals = jnp.moveaxis(result_vals, -1, axis)
        result_indices = jnp.moveaxis(result_indices, -1, axis)

    return result_vals, result_indices


@functools.partial(jax.jit, static_argnames=["i", "topk"])
def update_eagle_lists(
    i: int,
    score_list: jax.Array,
    token_list: jax.Array,
    parents_list: jax.Array,
    tree_info: tuple[jax.Array, jax.Array, jax.Array],
    topk: int,
):
    bs = score_list.shape[0]
    scores_update, tokens_update, parents_update = tree_info
    if i == 0:
        score_list = score_list.at[:bs, :1, :].set(scores_update[:bs])
        token_list = token_list.at[:bs, :topk].set(tokens_update[:bs])
        parents_list = parents_list.at[:bs, : topk + 1].set(parents_update[:bs])
    else:
        score_start = 1 + (i - 1) * topk
        token_start = topk + (i - 1) * topk * topk
        parent_start = topk + 1 + (i - 1) * topk

        score_list = score_list.at[:bs, score_start : score_start + topk, :].set(scores_update[:bs])
        token_list = token_list.at[:bs, token_start : token_start + topk * topk].set(
            tokens_update[:bs]
        )
        parents_list = parents_list.at[:bs, parent_start : parent_start + topk].set(
            parents_update[:bs]
        )
    return score_list, token_list, parents_list


# FIXME(pc) this should be jitted or convert as np.ndarray
# @functools.partial(jax.jit, static_argnames=["i"])
def update_forward_batch_info(
    forward_batch: ForwardBatch,
    i: int,
    input_ids: jax.Array,
    hidden_states: jax.Array,
    positions_base: jax.Array,
) -> ForwardBatch:
    forward_batch.input_ids = input_ids
    # FIXME(pc) hiddenstate will become NAN when forward path is very long, we still have no reason for this
    forward_batch.spec_info.hidden_states = hidden_states
    forward_batch.positions = positions_base + i
    return forward_batch


def select_top_k_tokens(
    i: int,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if i == 0:
        return select_top_k_tokens_step_0(topk_p, topk_index, hidden_states, scores, topk)
    else:
        return select_top_k_tokens_step_greater_0(
            jnp.asarray(i), topk_p, topk_index, hidden_states, scores, topk
        )


@functools.partial(jax.jit, static_argnames=["topk"])
def select_top_k_tokens_step_0(
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # The first step after extend
    input_ids = topk_index.flatten()
    hidden_states = jnp.repeat(hidden_states, topk, axis=0)
    scores = topk_p  # shape: (b, topk)
    tree_info = (
        jnp.expand_dims(topk_p, axis=1),  # shape: (b, 1, topk)
        topk_index,  # shape: (b, topk)
        jnp.tile(
            jnp.expand_dims(jnp.arange(-1, topk, dtype=jnp.float32), axis=0),
            (topk_p.shape[0], 1),
        ),  # shape: (b, topk + 1)
    )
    return input_ids, hidden_states, scores, tree_info


@functools.partial(jax.jit, static_argnames=["topk"])
def select_top_k_tokens_step_greater_0(
    i: jax.Array,
    topk_p: jax.Array,
    topk_index: jax.Array,
    hidden_states: jax.Array,
    scores: jax.Array,
    topk: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # The later decode steps
    expand_scores = jax.lax.mul(
        jnp.expand_dims(scores, axis=2), topk_p.reshape(-1, topk, topk)
    )  # (b, topk, 1) x (b, topk ,topk) -> (b, topk, topk)
    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.reshape(expand_scores.shape[0], -1), topk, axis=-1
    )  # (b, topk)
    scores = topk_cs_p  # shape: (b, topk)
    topk_index = topk_index.reshape(-1, topk**2)
    input_ids = jnp.take_along_axis(topk_index, topk_cs_index, axis=1).flatten()
    if hidden_states.shape[0] > 0:
        selected_input_index = topk_cs_index.flatten() // topk + jnp.repeat(
            jnp.arange(0, hidden_states.shape[0], topk), topk
        )
        hidden_states = hidden_states[selected_input_index, :]
    tree_info = (
        expand_scores,  # shape: (b, topk, topk)
        topk_index,  # shape: (b, topk * topk)
        topk_cs_index + (topk**2 * (i - 1) + topk),  # shape: (b, topk)
    )
    return input_ids, hidden_states, scores, tree_info
