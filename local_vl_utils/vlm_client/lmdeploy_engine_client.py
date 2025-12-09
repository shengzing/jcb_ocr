from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any, Sequence

from loguru import logger
from PIL import Image

from .base_client import (
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_USER_PROMPT,
    RequestError,
    SamplingParams,
    ServerError,
    VlmClient,
)
from .utils import aio_load_resource, gather_tasks, get_rgb_image, load_resource


class LmdeployEngineVlmClient(VlmClient):
    def __init__(
        self,
        lmdeploy_engine,  # lmdeploy.serve.vl_async_engine.VLAsyncEngine instance
        prompt: str = DEFAULT_USER_PROMPT,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        sampling_params: SamplingParams | None = None,
        text_before_image: bool = False,
        allow_truncated_content: bool = False,
        batch_size: int = 0,  # batch size for sync predict
        max_concurrency: int = 100,  # max concurrency for async predict
        use_tqdm: bool = True,
        debug: bool = False,
    ):
        super().__init__(
            prompt=prompt,
            system_prompt=system_prompt,
            sampling_params=sampling_params,
            text_before_image=text_before_image,
            allow_truncated_content=allow_truncated_content,
        )

        try:
            from lmdeploy import GenerationConfig
            from lmdeploy.serve.vl_async_engine import VLAsyncEngine
        except ImportError:
            raise ImportError("Please install lmdeploy to use LmdeployEngineVlmClient.")

        if not lmdeploy_engine:
            raise ValueError("lmdeploy_engine is None.")
        if not isinstance(lmdeploy_engine, VLAsyncEngine):
            raise ValueError(f"lmdeploy_engine must be an instance of {VLAsyncEngine}.")

        self.lmdeploy_engine = lmdeploy_engine
        self.model_max_length = lmdeploy_engine.session_len
        self.LmdeployGenerationConfig = GenerationConfig
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.use_tqdm = use_tqdm
        self.debug = debug
        self.session_id = 0
        self.session_id_lock = asyncio.Semaphore(1)

    def build_lmdeploy_generation_config(self, sampling_params: SamplingParams | None):
        sp = self.build_sampling_params(sampling_params)

        do_sample = ((sp.temperature or 0.0) > 0.0) and ((sp.top_k or 1) > 1)

        lmdeploy_sp_dict = {
            "temperature": sp.temperature,
            "top_p": sp.top_p,
            "top_k": sp.top_k,
            "repetition_penalty": sp.repetition_penalty,
            # WARNING - engine.py:606: num tokens is larger than max session len xxx. Update max_new_tokens=xxx.
            "max_new_tokens": sp.max_new_tokens if sp.max_new_tokens is not None else self.model_max_length,
        }

        return self.LmdeployGenerationConfig(
            **{k: v for k, v in lmdeploy_sp_dict.items() if v is not None},
            do_sample=do_sample,
            skip_special_tokens=False,
        )

    def predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        return self.batch_predict(
            [image],  # type: ignore
            [prompt],
            [sampling_params],
        )[0]

    def batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
    ) -> list[str]:
        if not isinstance(prompts, str):
            assert len(prompts) == len(images), "Length of prompts and images must match."
        if isinstance(sampling_params, Sequence):
            assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        if isinstance(priority, Sequence):
            assert len(priority) == len(images), "Length of priority and images must match."

        image_objs: list[Image.Image] = []
        for image in images:
            if isinstance(image, str):
                image = load_resource(image)
            if not isinstance(image, Image.Image):
                image = Image.open(BytesIO(image))
            image = get_rgb_image(image)
            image_objs.append(image)

        if isinstance(prompts, str):
            chat_prompts: list[str] = [prompts] * len(images)
        else:  # isinstance(prompts, Sequence[str])
            chat_prompts: list[str] = [prompt for prompt in prompts]

        if not isinstance(sampling_params, Sequence):
            gen_configs = [self.build_lmdeploy_generation_config(sampling_params)] * len(images)
        else:  # isinstance(sampling_params, Sequence)
            gen_configs = [self.build_lmdeploy_generation_config(sp) for sp in sampling_params]

        outputs = []
        batch_size = self.batch_size if self.batch_size > 0 else len(images)
        batch_size = max(1, batch_size)

        for i in range(0, len(images), batch_size):
            batch_image_objs = image_objs[i : i + batch_size]
            batch_chat_prompts = chat_prompts[i : i + batch_size]
            batch_gen_configs = gen_configs[i : i + batch_size]
            logger.debug(batch_chat_prompts)
            batch_outputs = self._predict_one_batch(
                batch_image_objs,
                batch_chat_prompts,
                batch_gen_configs,
            )
            outputs.extend(batch_outputs)

        return outputs

    def _predict_one_batch(
        self,
        image_objs: list[Image.Image],
        chat_prompts: list[str],
        gen_configs: list[Any],
    ):
        lmdeploy_prompts = list(zip(chat_prompts, image_objs))
        outputs = self.lmdeploy_engine.batch_infer(
            lmdeploy_prompts,  # type: ignore
            gen_config=gen_configs,
        )
        return [output.text for output in outputs]

    async def aio_predict(
        self,
        image: Image.Image | bytes | str,
        prompt: str = "",
        sampling_params: SamplingParams | None = None,
        priority: int | None = None,
    ) -> str:
        if isinstance(image, str):
            image = await aio_load_resource(image)
        if not isinstance(image, Image.Image):
            image = Image.open(BytesIO(image))
        image = get_rgb_image(image)

        lmdeploy_prompts = self.lmdeploy_engine._convert_prompts([(prompt, image)])[0]
        gen_config = self.build_lmdeploy_generation_config(sampling_params)

        async with self.session_id_lock:
            session_id = self.session_id
            self.session_id += 1

        generate_kwargs = {}
        if priority is not None:
            generate_kwargs["priority"] = priority

        response_parts = []
        async for output in self.lmdeploy_engine.generate(
                messages=lmdeploy_prompts,
                gen_config=gen_config,
                session_id=session_id,
                **generate_kwargs,
        ):
            if output.response is not None:
                response_parts.append(output.response)

        if not response_parts:  # this should not happen
            raise ServerError("No output from the server.")

        return "".join(response_parts)

    async def aio_batch_predict(
        self,
        images: Sequence[Image.Image | bytes | str],
        prompts: Sequence[str] | str = "",
        sampling_params: Sequence[SamplingParams | None] | SamplingParams | None = None,
        priority: Sequence[int | None] | int | None = None,
        semaphore: asyncio.Semaphore | None = None,
        use_tqdm=False,
        tqdm_desc: str | None = None,
    ) -> list[str]:
        if isinstance(prompts, str):
            prompts = [prompts] * len(images)
        if not isinstance(sampling_params, Sequence):
            sampling_params = [sampling_params] * len(images)
        if not isinstance(priority, Sequence):
            priority = [priority] * len(images)

        assert len(prompts) == len(images), "Length of prompts and images must match."
        assert len(sampling_params) == len(images), "Length of sampling_params and images must match."
        assert len(priority) == len(images), "Length of priority and images must match."

        if semaphore is None:
            semaphore = asyncio.Semaphore(self.max_concurrency)

        async def predict_with_semaphore(
            image: Image.Image | bytes | str,
            prompt: str,
            sampling_params: SamplingParams | None,
            priority: int | None,
        ):
            async with semaphore:
                return await self.aio_predict(
                    image=image,
                    prompt=prompt,
                    sampling_params=sampling_params,
                    priority=priority,
                )

        return await gather_tasks(
            tasks=[
                predict_with_semaphore(*args)
                for args in zip(
                    images,
                    prompts,
                    sampling_params,
                    priority,
                )
            ],
            use_tqdm=use_tqdm,
            tqdm_desc=tqdm_desc,
        )
