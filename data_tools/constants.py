TEXT_ENCODING = "utf-8"
HASH_CHUNK_BYTES = 1024 * 1024
SPLIT_SALT = "blind-assistant-p2"
DEFAULT_TEST_PERCENT = 20
PERCENT_BASE = 100
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
AUDIO_EXTENSIONS = frozenset({".wav", ".flac"})
IMAGE_FORMATS = frozenset({"JPEG", "PNG"})
JPEG_QUALITY = 95
JPEG_SUBSAMPLING = 0
METADATA_KEYS = ("xmp", "XML:com.adobe.xmp", "comment")
FACE_SCORE_THRESHOLD = 0.7
FACE_NMS_THRESHOLD = 0.3
FACE_TOP_K = 5000
FACE_INITIAL_SIZE = (320, 320)
FILE_COLUMN = "file"
GROUP_COLUMN = "group"
BATCH_COLUMN = "batch"
BASE_COLUMNS = (FILE_COLUMN, GROUP_COLUMN, BATCH_COLUMN)
DVC_BIN = "dvc"
CARD_FENCE = "---"
CARD_SEPARATOR = ":"
CARD_REQUIRED_FIELDS = (
    "dataset",
    "task",
    "source",
    "collected_on",
    "license",
    "consent",
    "collected_by",
)
LOCK_SALT_KEY = "salt"
LOCK_PERCENT_KEY = "testPercent"
LOCK_ENTRIES_KEY = "entries"
LOCK_JSON_INDENT = 1
