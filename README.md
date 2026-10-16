# async-parity

Find sync/async method pairs that have drifted apart.

Python libraries often ship a method and its coroutine twin side by side: `run` / `run_async`, `chat` / `achat`, `embed` / `aembed`. They start as copies. Then a bug gets fixed on one side only. `async-parity` normalizes both bodies and shows what's left of the difference:

- drops `await`, `async for` and `async with`;
- drops docstrings;
- maps counterpart names (`achat` → `chat`, `run_async` → `run`, `AsyncOpenAI` → `OpenAI`).

It pairs methods inside a class (`x` with `x_async`, `ax`, `a_x`, `async_x`, and private `_x` with `_ax`), and same-named methods of twin classes such as `Client` / `AsyncClient` or `Client` / `ClientAsync`, even when they live in different modules.

The remaining differences are either deliberate or a parity bug.

## Example

This is the shape of a real bug, [deepset-ai/haystack#12664](https://github.com/deepset-ai/haystack/pull/12664). The sync pipeline lets a nested `PipelineRuntimeError` through unchanged, while the async path wraps it a second time and loses the original context:

```python
class Pipeline:
    def _run_component(self, instance, inputs):
        try:
            return instance.run(**inputs)
        except PipelineRuntimeError as error:
            raise error
        except Exception as error:
            raise PipelineRuntimeError.from_exception(error) from error

    async def _run_component_async(self, instance, inputs):
        try:
            return await instance.run_async(**inputs)
        except Exception as error:
            raise PipelineRuntimeError.from_exception(error) from error
```

Run on the real `haystack/core/pipeline/pipeline.py` from just before the fix, it reports the missing clauses. The async side names its variables differently, so a few renamed lines come along with them (output trimmed):

```console
$ async-parity haystack/core/pipeline/pipeline.py
### haystack/core/pipeline/pipeline.py:Pipeline._run_component vs _run_component_async
--- _run_component
+++ _run_component_async
     try:
-        component_output = instance.run(**inputs_copy)
-    except BreakpointException as error:
-        raise error
-    except PipelineRuntimeError as runtime_error:
-        raise runtime_error
+        outputs = _execute_component_async(instance, **component_inputs_copy)
     except Exception as error:
…
```

After the fix, those `except` lines no longer show up in the report.

## Bugs it has found

Found with this approach and reported upstream. Every row was confirmed with a reproduction before it was reported.

| Project | Link | Status | What diverged |
|---|---|---|---|
| Haystack | [#12664](https://github.com/deepset-ai/haystack/pull/12664) | merged | async pipeline re-wrapped nested errors |
| Haystack | [#12662](https://github.com/deepset-ai/haystack/pull/12662) | merged | `max_workers` ignored by the multi-retrievers' `run_async` |
| Haystack | [#12663](https://github.com/deepset-ai/haystack/pull/12663) | merged | `encoding_format="float"` missing in `OpenAIDocumentEmbedder.run_async` |
| Haystack | [#12358](https://github.com/deepset-ai/haystack/pull/12358) | merged | `EmbeddingBasedDocumentSplitter.run_async` fell back to the sync path |
| haystack-core-integrations | [#3926](https://github.com/deepset-ai/haystack-core-integrations/pull/3926) | merged | `CohereDocumentEmbedder.run_async` didn't batch like the sync path |
| chatlas | [#455](https://github.com/posit-dev/chatlas/pull/455) | merged | sync generator tools returned the generator object in async chats |
| OpenInference | [#4066](https://github.com/Arize-ai/openinference/pull/4066) | open | Agent Spec `*_async` span methods raised `NotImplementedError` |
| langchain-aws | [#1395](https://github.com/langchain-ai/langchain-aws/pull/1395) | open | `BedrockLLM` async streaming lost or broke output for several providers |
| langchain-aws | [#1396](https://github.com/langchain-ai/langchain-aws/pull/1396) | open | `BedrockLLM._acall` refused the default `streaming=False` (private twins) |
| pymilvus | [#3863](https://github.com/milvus-io/pymilvus/pull/3863) | open | `AsyncGrpcHandler` file resource calls failed on any extra kwarg (twin classes) |
| agno | [#11123](https://github.com/agno-agi/agno/issues/11123) | issue | `async_upsert` deleted old chunks before embedding in 4 vector stores |
| agno | [#11124](https://github.com/agno-agi/agno/issues/11124) | issue | `Knowledge.asearch` skipped rerankers whose `arerank` has no `limit` |
| llama_index | [#23549](https://github.com/run-llama/llama_index/issues/23549) | issue | Azure Inference `astream_complete` passed `stream` twice |
| llama_index | [#23550](https://github.com/run-llama/llama_index/issues/23550) | issue | HuggingFace API async client closed after one stream |
| llama_index | [#23552](https://github.com/run-llama/llama_index/issues/23552) | issue | `OpenAIResponses.achat` dropped reasoning items (private twins) |

## Install and use

```console
pip install git+https://github.com/pcbeingused333/async-parity
async-parity path/to/package            # pairs where the sync side does something the async side doesn't
async-parity path/to/package --summary  # one line per pair
async-parity path/to/package --all      # every divergent pair
```

Test directories are skipped unless you pass `--include-tests`. The exit code is 1 when something is reported, so it can run in CI.

## Limitations

- Pairing is by name only. Twin classes must be named `Async<Name>` or `<Name>Async`, and when two modules define the same class name, the first one scanned wins.
- Generated code (sync clients produced from async ones with tools like `unasync`) never diverges, so scanning it is wasted time.
- Most reports are deliberate differences: `asyncio.to_thread`, a different client object, an async-only lock. It's a reading list, not a verdict. The default filter (only show pairs where the sync side has something the async side lacks) cuts most of the noise.
- Thin wrappers that delegate to different helpers (`return _impl.run(...)` vs `return _impl.arun(...)`) are reported when the helper names don't follow the counterpart conventions.

## License

MIT
