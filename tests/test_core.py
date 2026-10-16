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
