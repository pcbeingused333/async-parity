from pathlib import Path
from textwrap import dedent

from async_parity import scan_path, scan_source
from async_parity.cli import main


def divergences(source: str):
    return list(scan_source(dedent(source)))


def test_identical_pair_after_stripping_async_is_not_reported():
    assert not divergences(
        """
        class Client:
            def chat(self, messages):
                payload = self._build(messages)
                return self._client.send(payload)

            async def achat(self, messages):
                payload = self._build(messages)
                return await self._aclient.asend(payload)
        """
    )


def test_counterpart_names_are_mapped_but_ordinary_a_names_are_not():
    # `append` and `args` start with "a" but have no sync twin, so they must stay as they are.
    assert not divergences(
        """
        class Store:
            def write_docs(self, docs, args):
                out = []
                for d in docs:
                    out.append(self.embed(d, args))
                return out

            async def write_docs_async(self, docs, args):
                out = []
                for d in docs:
                    out.append(await self.aembed(d, args))
                return out
        """
    )


def test_async_client_class_maps_to_sync_class():
    assert not divergences(
        """
        class Provider:
            def client(self):
                return OpenAI(api_key=self.key)

            async def aclient(self):
                return AsyncOpenAI(api_key=self.key)
        """
    )


def test_except_clause_missing_on_async_side_is_reported():
    # The shape of deepset-ai/haystack#12664: the sync path lets a nested pipeline error through,
    # the async path re-wraps it.
    found = divergences(
        """
        class Pipeline:
            def run_component(self, instance, inputs):
                try:
                    return instance.run(**inputs)
                except PipelineRuntimeError as error:
                    raise error
                except Exception as error:
                    raise PipelineRuntimeError.from_exception(error) from error

            async def run_component_async(self, instance, inputs):
                try:
                    return await instance.run_async(**inputs)
                except Exception as error:
                    raise PipelineRuntimeError.from_exception(error) from error
        """
    )
    assert len(found) == 1
    (div,) = found
    assert (div.sync_name, div.async_name) == ("run_component", "run_component_async")
    assert any("PipelineRuntimeError as error" in line for line in div.sync_only)


def test_argument_dropped_on_async_side_is_reported():
    (div,) = divergences(
        """
        class Embedder:
            def embed(self, texts):
                return self.client.create(input=texts, encoding_format="float")

            async def aembed(self, texts):
                return await self.client.acreate(input=texts)
        """
    )
    assert div.sync_only == ["return self.client.create(input=texts, encoding_format='float')"]


def test_private_twins_are_paired():
    # The shape of langchain-ai/langchain-aws#1396: `_call` handles the default case, `_acall`
    # refuses it.
    (div,) = divergences(
        """
        class BedrockLLM:
            def _call(self, prompt):
                if not self.streaming:
                    return self._invoke(prompt)
                return self._stream_text(prompt)

            async def _acall(self, prompt):
                if not self.streaming:
                    raise ValueError("Streaming must be set to True for async operations.")
                return await self._astream_text(prompt)
        """
    )
    assert (div.sync_name, div.async_name) == ("_call", "_acall")
    assert [line.strip() for line in div.sync_only] == ["return self._invoke(prompt)"]


def test_dunder_methods_are_not_paired_as_private_twins():
    assert not divergences(
        """
        class Stream:
            def __iter__(self):
                return iter(self.items)

            def __aiter__(self):
                return self
        """
    )


def test_twin_classes_are_compared_method_by_method():
    # The shape of milvus-io/pymilvus#3863: the async handler unpacks the caller's kwargs into a
    # helper that only takes the call context.
    (div,) = divergences(
        """
        class GrpcHandler:
            def add_file_resource(self, name, path, timeout=None, **kwargs):
                req = Prepare.add_file_resource(name=name, path=path)
                resp = self._stub.AddFileResource(req, timeout=timeout, metadata=_api_level_md(kwargs.get("context")))
                check_status(resp)

            def list_file_resources(self, timeout=None, **kwargs):
                return self._stub.ListFileResources(Prepare.list_file_resources(), timeout=timeout)


        class AsyncGrpcHandler:
            async def add_file_resource(self, name, path, timeout=None, **kwargs):
                req = Prepare.add_file_resource(name=name, path=path)
                resp = await self._async_stub.AddFileResource(req, timeout=timeout, metadata=_api_level_md(**kwargs))
                check_status(resp)

            async def list_file_resources(self, timeout=None, **kwargs):
                return await self._async_stub.ListFileResources(Prepare.list_file_resources(), timeout=timeout)
        """
    )
    assert div.header() == "<string>:GrpcHandler.add_file_resource vs AsyncGrpcHandler.add_file_resource"
    assert any("kwargs.get('context')" in line for line in div.sync_only)


def test_twin_classes_in_different_files_are_compared(tmp_path: Path):
    (tmp_path / "sync_client.py").write_text(
        dedent(
            """
            class Client:
                def search(self, query, limit=10):
                    return self._http.get("/search", params={"q": query, "limit": limit})
            """
        )
    )
    (tmp_path / "async_client.py").write_text(
        dedent(
            """
            class AsyncClient:
                async def search(self, query, limit=10):
                    return await self._http.get("/search", params={"q": query})
            """
        )
    )
    (div,) = scan_path(tmp_path)
    assert (div.cls, div.async_name) == ("Client", "AsyncClient.search")
    assert div.path == tmp_path / "sync_client.py"


def test_docstrings_are_ignored():
    assert not divergences(
        '''
        class A:
            def run(self):
                """Sync docs."""
                return 1

            async def run_async(self):
                """Async docs, worded differently."""
                return 1
        '''
    )


def test_scan_path_skips_excluded_directories(tmp_path: Path):
    bad = dedent(
        """
        class C:
            def go(self):
                self.a()
                self.b()

            async def ago(self):
                await self.aa()
        """
    )
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "pkg" / "mod.py").write_text(bad)
    (tmp_path / "tests" / "test_mod.py").write_text(bad)

    found = list(scan_path(tmp_path, exclude=("tests",)))

    assert [d.path.parent.name for d in found] == ["pkg"]


def test_cli_exit_code_and_summary(tmp_path: Path, capsys):
    (tmp_path / "ok.py").write_text(
        "class C:\n    def run(self):\n        return 1\n    async def arun(self):\n        return 1\n"
    )
    assert main([str(tmp_path), "--summary"]) == 0

    (tmp_path / "bad.py").write_text(
        "class C:\n    def run(self):\n        self.x()\n        return 1\n    async def arun(self):\n        return 1\n"
    )
    assert main([str(tmp_path), "--summary"]) == 1
    assert "bad.py:C.run vs arun" in capsys.readouterr().out


def test_cli_pairs_twin_classes_given_as_separate_files(tmp_path: Path, capsys):
    sync_file, async_file = tmp_path / "client.py", tmp_path / "aclient.py"
    sync_file.write_text(
        "class Client:\n    def get(self, key):\n        return self._db.get(key, consistent=True)\n"
    )
    async_file.write_text(
        "class AsyncClient:\n    async def get(self, key):\n        return await self._db.get(key)\n"
    )
    assert main([str(sync_file), str(async_file), "--summary"]) == 1
    assert "Client.get vs AsyncClient.get" in capsys.readouterr().out
