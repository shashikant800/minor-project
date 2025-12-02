from pathlib import Path

from nltk import Tree
import argparse
import pickle


def factorize(tree):
    def track(tree, i):
        label = tree.label()
        if len(tree) == 1 and not isinstance(tree[0], Tree):
            return (i+1 if label is not None else i), []
        j, spans = i, []
        for child in tree:
            j, s = track(child, j)
            spans += s
        if label is not None and j > i:
            spans = [[i, j, label]] + spans
        elif j > i:
            spans = [[i, j, 'NULL']] + spans
        return j, spans
    return track(tree, 0)[1]


def _load_prebuilt_dataset(file_path):
    """Return the dataset if the given file is already a pickle with the expected structure."""
    expected_keys = {'word', 'pos', 'gold_tree'}
    try:
        with file_path.open('rb') as f:
            payload = pickle.load(f)
    except Exception as exc:
        raise ValueError(f"Failed to unpickle dataset from '{file_path}': {exc}") from exc

    if isinstance(payload, dict) and expected_keys.issubset(payload.keys()):
        print(f"[INFO] '{file_path}' already contains preprocessed data; returning as-is.")
        return payload

    raise ValueError(
        f"File '{file_path}' appears to be a pickle but does not contain the expected keys "
        f"{sorted(expected_keys)}."
    )


def create_dataset(file_name):
    file_path = Path(file_name)
    if file_path.suffix in ('.pickle', '.pkl'):
        return _load_prebuilt_dataset(file_path)

    word_array = []
    pos_array = []
    gold_trees = []
    skipped = 0
    # Force UTF-8 decoding but keep going even if the source file mixes encodings.
    with open(file_name, 'r', encoding='utf-8', errors='replace') as f:
        for idx, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                tree = Tree.fromstring(line)
            except ValueError as exc:
                skipped += 1
                # Surface a few examples to help debug corrupted inputs.
                if skipped <= 5:
                    print(f"[WARN] skipping malformed tree at line {idx}: {exc}")
                continue
            token = tree.pos()
            word, pos = zip(*token)
            word_array.append(word)
            pos_array.append(pos)
            gold_trees.append(factorize(tree))

    if skipped:
        print(f"[INFO] Skipped {skipped} malformed lines while reading {file_name}")

    return {'word': word_array,
            'pos': pos_array,
            'gold_tree':gold_trees}





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='preprocess ptb file.'
    )
    parser.add_argument('--train_file', default='data/ptb-train.txt')
    parser.add_argument('--val_file', default='data/ptb-valid.txt')
    parser.add_argument('--test_file', default='data/ptb-test.txt')
    parser.add_argument('--cache_path', default='data/')

    args = parser.parse_args()
    cache_dir = Path(args.cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    result = create_dataset(args.train_file)
    with open(cache_dir / "train.pickle", "wb") as f:
        pickle.dump(result, f)

    result = create_dataset(args.val_file)
    with open(cache_dir / "val.pickle", "wb") as f:
        pickle.dump(result, f)

    result = create_dataset(args.test_file)
    with open(cache_dir / "test.pickle", "wb") as f:
        pickle.dump(result, f)











