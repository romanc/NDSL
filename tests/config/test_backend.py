import pytest

from ndsl import Backend


def test_backend_building() -> None:
    Backend("st:python:cpu:IJK")
    Backend("st:numpy:cpu:IJK")
    Backend("st:gt:cpu:IJK")
    Backend("st:gt:cpu:KJI")
    Backend("st:gt:gpu:KJI")
    Backend("st:dace:cpu:IJK")
    Backend("orch:dace:cpu:IJK")
    Backend("st:dace:cpu:KIJ")
    Backend("orch:dace:cpu:KIJ")
    Backend("st:dace:cpu:KJI")
    Backend("orch:dace:cpu:KJI")
    Backend("st:dace:gpu:KJI")
    Backend("orch:dace:gpu:KJI")

    unknown_backend = "bad:name:good:number"
    with pytest.raises(ValueError, match=f"Unknown {unknown_backend}, options are .*"):
        Backend(unknown_backend)


def test_ill_formed_backend_raises() -> None:
    from ndsl.config.backend import _NDSL_TO_GT4PY_BACKEND_NAMING

    # push an ill-formed backend into _NDSL_TO_GT4PY_BACKEND_NAMING for testing
    ill_formed_backend = "ill:formed"
    _NDSL_TO_GT4PY_BACKEND_NAMING[ill_formed_backend] = "non-existing"
    with pytest.raises(
        ValueError, match=f"Backend {ill_formed_backend} is ill-formed."
    ):
        Backend(ill_formed_backend)


def test_non_existing_gt4py_backend_raises() -> None:
    from ndsl.config.backend import _NDSL_TO_GT4PY_BACKEND_NAMING

    # push a non existing gt4py backend into _NDSL_TO_GT4PY_BACKEND_NAMING for testing
    plausible_backend = "orch:python:cpu:IJK"
    non_existing = "dace:nope"
    _NDSL_TO_GT4PY_BACKEND_NAMING[plausible_backend] = non_existing
    with pytest.raises(
        ValueError,
        match=f"NDSL backend {plausible_backend} does not have a working GT4Py version. GT4Py backend {non_existing} is not registered.",
    ):
        Backend(plausible_backend)


def test_requesting_gpu_backend_on_cpu_raises() -> None:
    from ndsl.config.backend import _NDSL_TO_GT4PY_BACKEND_NAMING

    # push an inconsistent backend into _NDSL_TO_GT4PY_BACKEND_NAMING for testing
    inconsistent_backend = "st:python:gpu:IJK"  # debug backend on GPU
    _NDSL_TO_GT4PY_BACKEND_NAMING[inconsistent_backend] = "debug"
    with pytest.raises(ValueError, match="NDSL backend requested .* targets GPU"):
        Backend(inconsistent_backend)


def test_backend_operators() -> None:
    backend_A = Backend("st:numpy:cpu:IJK")
    backend_B = Backend("st:numpy:cpu:IJK")

    assert backend_A == backend_B
    assert not (backend_A != backend_B)

    with pytest.raises(
        NotImplementedError,
        match="Backend equality operator for .* is not implemented.",
    ):
        is_dictionary = backend_A == dict()


def test_as_safe_for_path() -> None:
    assert Backend.python().as_safe_for_path() == "st_python_cpu_IJK"


def test_is_stencil() -> None:
    assert Backend.python().is_stencil()
    assert not Backend("orch:dace:cpu:IJK").is_stencil()


def test_is_gpu() -> None:
    assert Backend.gpu().is_gpu_backend()
    assert not Backend.cpu().is_gpu_backend()
