# async-parity

Find sync/async method pairs that have drifted apart.

Python libraries often ship a method and its coroutine twin side by side: `run` / `run_async`, `chat` / `achat`, `embed` / `aembed`. They start as copies. Then a bug gets fixed on one side only. `async-parity` normalizes both bodies and shows what's left of the difference:

- drops `await`, `async for` and `async with`;
- drops docstrings;
- maps counterpart names (`achat` → `chat`, `run_async` → `run`, `AsyncOpenAI` → `OpenAI`).

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

These were found with this approach and fixed upstream:

| Project | PR | What diverged |
|---|---|---|
| Haystack | [#12664](https://github.com/deepset-ai/haystack/pull/12664) | async pipeline re-wrapped nested errors |
| Haystack | [#12662](https://github.com/deepset-ai/haystack/pull/12662) | `max_workers` ignored by the multi-retrievers' `run_async` |
| Haystack | [#12663](https://github.com/deepset-ai/haystack/pull/12663) | `encoding_format="float"` missing in `OpenAIDocumentEmbedder.run_async` |
| Haystack | [#12358](https://github.com/deepset-ai/haystack/pull/12358) | `EmbeddingBasedDocumentSplitter.run_async` fell back to the sync path |
| haystack-core-integrations | [#3926](https://github.com/deepset-ai/haystack-core-integrations/pull/3926) | `CohereDocumentEmbedder.run_async` didn't batch like the sync path |

## Install and use

```console
pip install git+https://github.com/<you>/async-parity
async-parity path/to/package            # pairs where the sync side does something the async side doesn't
async-parity path/to/package --summary  # one line per pair
async-parity path/to/package --all      # every divergent pair
```

Test directories are skipped unless you pass `--include-tests`. The exit code is 1 when something is reported, so it can run in CI.

## Limitations

- It only pairs methods within the same class, by name (`x` with `x_async`, `ax`, `a_x` or `async_x`).
- Most reports are deliberate differences: `asyncio.to_thread`, a different client object, an async-only lock. It's a reading list, not a verdict. The default filter (only show pairs where the sync side has something the async side lacks) cuts most of the noise.
- Thin wrappers that delegate to different helpers (`return _impl.run(...)` vs `return _impl.arun(...)`) are reported when the helper names don't follow the counterpart conventions.

## License

MIT
