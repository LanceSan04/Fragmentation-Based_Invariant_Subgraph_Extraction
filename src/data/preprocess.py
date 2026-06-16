"""Dataset preprocessing coordinator.

The load-convert-save pattern is based on Group Graph's data_loader.py.
Dual-view graph construction, metadata, splitting, and cache validation are
project-specific and intentionally remain unimplemented.
"""


def load_raw_records(config):
    raise NotImplementedError


def process_record(record, config):
    raise NotImplementedError


def process_split(records, config):
    raise NotImplementedError


def build_metadata(config):
    raise NotImplementedError


def save_processed_split(examples, metadata, output_path):
    raise NotImplementedError


def validate_processed_cache(path, config):
    raise NotImplementedError


def run_preprocessing(config):
    raise NotImplementedError
