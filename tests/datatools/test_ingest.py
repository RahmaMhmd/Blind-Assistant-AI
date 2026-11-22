from data_tools.checks import check_dataset
from data_tools.ingest import ingest_batch
from data_tools.manifest import read_manifest
from data_tools.problems import ProblemCode
from data_tools.registry import get_spec
from tests.datatools.support import (
    DATASET,
    FakeDetector,
    make_image,
    make_layout,
    name_of,
)

GROUPS = 30


def _batch(tmp_path, count=GROUPS):
    source = tmp_path / "incoming"
    for group in range(count):
        make_image(source / name_of(group), shade=group * 3, tagged=True)

    return source


def _setup(tmp_path):
    layout = make_layout(tmp_path)

    return layout, get_spec(layout, DATASET)


def _codes(problems):
    return [problem.code for problem in problems]


def test_ingest_cleans_and_lists_every_file(tmp_path):
    layout, spec = _setup(tmp_path)

    report = ingest_batch(layout, spec, _batch(tmp_path), None)

    assert len(report.added) == GROUPS
    assert report.rejected == []
    assert len(read_manifest(layout.manifest(DATASET))) == GROUPS
    assert ProblemCode.HAS_METADATA not in _codes(check_dataset(layout, spec))


def test_ingest_rejects_bad_files(tmp_path):
    layout, spec = _setup(tmp_path)
    source = _batch(tmp_path, count=1)
    make_image(source / "IMG_1.jpg")
    make_image(source / "other_b01_g001_001.jpg")
    (source / name_of(7)).write_text("broken")
    (source / name_of(8, ext=".bmp")).write_bytes(b"x")

    report = ingest_batch(layout, spec, source, None)

    assert _codes(report.rejected) == [
        ProblemCode.BAD_NAME,
        ProblemCode.UNREADABLE,
        ProblemCode.BAD_EXTENSION,
        ProblemCode.WRONG_DATASET,
    ]


def test_ingest_drops_images_with_faces(tmp_path):
    layout, spec = _setup(tmp_path)
    source = _batch(tmp_path, count=2)
    detector = FakeDetector({name_of(1)})

    report = ingest_batch(layout, spec, source, detector)

    assert report.added == [name_of(0)]
    assert _codes(report.rejected) == [ProblemCode.FACES_FOUND]
    assert not (layout.dataset_dir(DATASET) / name_of(1)).exists()


def test_ingest_skips_known_names_and_content(tmp_path):
    layout, spec = _setup(tmp_path)
    source = _batch(tmp_path, count=2)
    ingest_batch(layout, spec, source, None)
    make_image(source / name_of(9), shade=0, tagged=True)

    report = ingest_batch(layout, spec, source, None)

    assert report.added == []
    assert _codes(report.rejected) == [ProblemCode.DUPLICATE] * 3
