"""
This module contains a collection of functions used to generate different
versions of the data set.
Todo:

"""

import os
import re
from glob import glob
import socket
from itertools import chain
from munch import Munch
import numpy as np
from . import transforms
import joblib

data_dir_path = os.path.join(os.path.dirname(__file__), "data")
_babi_1K_path = os.path.join(data_dir_path, "tasks_1-20_v1-2", "en")
_babi_10K_path = os.path.join(data_dir_path, "tasks_1-20_v1-2", "en-10k")
joblib_cache_dir_path = os.path.join(
    data_dir_path, f"joblib_cache_{socket.gethostname()}"
)
memory = joblib.Memory(cachedir=joblib_cache_dir_path)


def _read_babi_lines(path):
    story_lines = []
    with open(path) as f:
        for line in f:
            line_num = int(line.split(" ")[0])
            if line_num == 1 and story_lines:
                story_lines = []
            if "?" in line:
                yield story_lines, line
            else:
                story_lines.append(line)


def load_task(task_num, version, folder_path):
    """
    Creates a set of QA tuples for a given task.
    """
    glob_path = os.path.join(folder_path, f"qa{task_num}_*{version}*")
    task_path = glob(glob_path)[0]

    output = []

    for story_lines, question_line in _read_babi_lines(task_path):
        # The hint line numbers include the question lines, which I'm
        # removing. So I here I match the hint to the line number and
        # remap to a 0 indexed value.
        story_line_indices = [re.match("^\d+", line).group(0) for line in story_lines]

        # hint = re.search(r'\d+$', question_line)
        # hint = hint.group(0)
        # hint = story_line_indices.index(hint)

        hints = re.findall(f"\d+", question_line)
        hints = hints[1:]  # First is line number.
        hints = [story_line_indices.index(hint) for hint in hints]

        story = "".join(story_lines)
        story = story.replace("\n", " ")
        story = re.sub(r"\d+\s", r"", story)

        question, answer, _ = question_line.split("\t")
        question = re.sub(r"\d+\s", r"", question)
        output.append((story, question, answer, hints))

    return output


def load_tasks(version, folder_path, task_subset=None):
    """
    Load the QA tuples for all tasks.
    """
    output = []
    for i in range(1, 21):
        if task_subset is not None and i not in task_subset:
            continue
        output.append(load_task(i, version, folder_path))
    return output


def get_embeddings(texts, min_vocab_size=0):
    """
    Build emeddding mappings based on `texts`.
    Args:
        texts (Iterable[str]): A sequence of text to fit the embeddings
            on.
        min_vocab_size (Optional[int]): Minimum number of words, not
            counting those in `texts` to include in the embedding
            vocabulary.
    Returns:
        (tuple): tuple containing:
            word_to_vec (dict): A map between word tokens and numpy vectors.
            word_to_id (dict): A map between word tokens and embedding ids.
            embedding_matrix (np.ndarry): A numpy matrix that maps between
                embedding ids and embedding vectors.
    """
    data_vocab = set()
    for token_list in transforms.token_strs_to_token_lists(texts):
        for token in token_list:
            data_vocab.add(token)

    word_to_vec = {}
    embeddings_path = os.path.join(data_dir_path, "glove.6B/glove.6B.200d.txt")
    with open(embeddings_path) as f:
        for line_num, line in enumerate(f):
            values = line.split()
            word = values[0]
            if min_vocab_size < line_num + 1 and word not in data_vocab:
                continue
            vector = np.asarray(values[1:], dtype="float32")
            word_to_vec[word] = vector

    # TODO: Variables redundant and unclear. Total vocab should mean
    # total vocab.
    total_vocab = data_vocab | set(word_to_vec.keys())
    rand_state = np.random.RandomState(42)
    word_to_id = {"<PAD>": 0, "<EOS>": 1, "<START>": 2}
    num_meta_tokens = 3
    embedding_matrix = rand_state.rand(len(total_vocab) + num_meta_tokens, 200)
    embedding_matrix[0] = 0
    embedding_matrix[1] = 1
    embedding_matrix[2] = 2
    total_vocab = sorted(total_vocab)  # Sort for consistent ids.
    for i, word in enumerate(total_vocab):
        word_id = i + num_meta_tokens
        if word in word_to_vec:
            embedding_matrix[word_id] = word_to_vec[word]
        word_to_id[word] = word_id

    return word_to_vec, word_to_id, embedding_matrix


def ids_to_text(ids, id_to_word):
    tokens = [id_to_word[id] for id in ids]
    return " ".join(tokens)


def id_lists_to_texts(id_lists, id_to_word):
    return [ids_to_text(ids, id_to_word) for ids in id_lists]


def answer_ids_to_text(ids, id_to_word):
    text = ids_to_text(ids, id_to_word)
    text = text.split("<EOS>")[0]
    return text.strip()


def answer_id_lists_to_texts(id_lists, id_to_word):
    return [answer_ids_to_text(ids, id_to_word) for ids in id_lists]


def get_train_val_test(train_sqas, test_sqas, task_subset=None):
    if task_subset is None:
        task_subset = range(1, 21)

    flat_train_sqas = chain(*train_sqas)
    flat_val_sqas = []
    flat_test_sqas = []
    test_sqas = np.array(test_sqas, dtype=object)  # Don't coerce types.

    for task_sqas, task_num in zip(test_sqas, task_subset):
        _, val_indices, test_indices = get_train_val_test_indices(
            len(task_sqas), val_ratio=0.50, test_ratio=0.50, seed=42 + task_num
        )
        flat_val_sqas.extend(task_sqas[val_indices])
        flat_test_sqas.extend(task_sqas[test_indices])

    def process_sqas(sqas):
        stories, questions, awnswers, hints = zip(*sqas)
        stories, questions, awnswers = map(
            transforms.tokenize_texts, [stories, questions, awnswers]
        )
        token_sqas = tuple(map(np.array, [stories, questions, awnswers, hints]))
        return token_sqas

    train_stories, train_questions, train_answers, train_hints = process_sqas(
        flat_train_sqas
    )
    val_stories, val_questions, val_answers, val_hints = process_sqas(flat_val_sqas)
    test_stories, test_questions, test_answers, test_hints = process_sqas(
        flat_test_sqas
    )

    data = Munch(
        X_train_stories=train_stories,
        X_train_questions=train_questions,
        train_hints=train_hints,
        y_train=train_answers,
        X_val_stories=val_stories,
        X_val_questions=val_questions,
        val_hints=val_hints,
        y_val=val_answers,
        X_test_stories=test_stories,
        X_test_questions=test_questions,
        test_hints=test_hints,
        y_test=test_answers,
    )

    return data


def texts_to_ids(texts, word_to_id, max_sequence_length=None):
    """
    Args:
        texts (Iterable[str]): A sequence of texts to fit the embeddings
            on.
        word_to_id (dict): A map between word tokens and embedding ids.
        max_sequence_length (Optional[int]): The maximum number of words
            to include in each line of dialogue. Shorter sequences will
            be padded with the <PAD> vector.
    Returns:
        X (np.ndarray): An array with shape `(len(texts), max_sequence_length)`
            containing the correct embedding ids.
    """

    token_lists = transforms.token_strs_to_token_lists(texts)
    texts_ids = []
    for token_list in token_lists:
        word_ids = []
        for token in token_list:
            if token in word_to_id:
                word_ids.append(word_to_id[token])
        # Add <EOS> token.
        word_ids.append(1)
        texts_ids.append(word_ids)

    if max_sequence_length is None:
        max_sequence_length = max(len(ids) for ids in texts_ids)

    X = np.zeros((len(texts), max_sequence_length), dtype=int)
    for i, text_ids in enumerate(texts_ids):
        text_ids = text_ids[:max_sequence_length]
        X[i, : len(text_ids)] = text_ids

    return X


def align_padding(*id_matrices, forced_length=None):
    if forced_length is None:
        max_row_len = max(matrix.shape[1] for matrix in id_matrices)
    else:
        max_row_len = forced_length
    output = []
    for matrix in id_matrices:
        padding_needed = max_row_len - matrix.shape[1]
        padded_matrix = np.pad(
            matrix, ((0, 0), (0, padding_needed)), "constant", constant_values=(0, 0)
        )
        output.append(padded_matrix)
    return output


def get_time_shifted(sequences):
    # TODO: Start, pad, etc. should be constants.
    start_tokens = np.zeros((sequences.shape[0], 1)) + 2  # <START> == 2
    return np.concatenate([start_tokens, sequences[:, :-1]], axis=1)


def get_train_val_test_indices(num_rows, val_ratio=0.25, test_ratio=0.25, seed=42):
    """Return indices of the train, test, and validation sets."""
    rand = np.random.RandomState(seed)
    indices = rand.permutation(range(num_rows))
    train_ratio = 1 - val_ratio - test_ratio
    train_indices = indices[: int(len(indices) * train_ratio)]
    val_indices = indices[
        len(train_indices) : len(train_indices) + int(len(indices) * val_ratio)
    ]
    test_indices = indices[len(train_indices) + len(val_indices) :]
    return train_indices, val_indices, test_indices


def get_sentence_masks(sentences, period_symbol):
    mask = np.zeros(sentences.shape)
    mask[sentences == period_symbol] = 1
    return mask


def get_hint_masks(stories, hints, period_symbol):
    masks = []
    for story, story_hints in zip(stories, hints):
        mask = []
        sentence_num = 0
        for word in story:
            # WARNING: For reasons I don't understand, positively matching
            # the meta-tokens helps it converge faster. It still does okay
            # if you remove 0, but <EOS> bugs it. And I sort of think the 0
            # does help. Maybe it's less attention than ignore...
            if sentence_num in story_hints or word in (0, 1, 2):  # Meta-tokens.
                mask.append(1)
            else:
                mask.append(0)
            if word == period_symbol:
                sentence_num += 1
        masks.append(mask)
    masks = np.array(masks)

    assert masks.shape == stories.shape

    return masks


@memory.cache
def get_babi_embeddings(use_10k=False, min_vocab_size=0):
    # Calculate embedding matrix with suitable vocab for all
    # tasks. I don't calculate a task specific vocab because
    # I want the word indices to be the same in all task data
    # sets.
    babi_path = _babi_10K_path if use_10k else _babi_1K_path
    train_tasks = load_tasks("train", babi_path)
    test_tasks = load_tasks("test", babi_path)

    # Don't tokenize the hints.
    train_tasks = [t[:-1] for t in chain(*train_tasks)]
    test_tasks = [t[:-1] for t in chain(*test_tasks)]

    tokenized_texts = map(transforms.to_tokens, chain(*chain(train_tasks, test_tasks)))
    return get_embeddings(tokenized_texts, min_vocab_size=min_vocab_size)


def get_story_sents(stories, period_symbol):
    output = []
    for story in stories:
        sents = []
        cur_sent = []
        for word in story:
            cur_sent.append(word)
            if word == period_symbol:
                cur_sent.append(1)  # <EOS>
                sents.append(cur_sent)
                cur_sent = []
        output.append(sents)
    return output


def align_story_sents(*story_sets, num_sents=None, sent_length=None):
    if num_sents is None or sent_length is None:
        all_stories = list(chain(*story_sets))
        all_sents = list(chain(*all_stories))
    if num_sents is None:
        num_sents = max(len(story) for story in all_stories)
    if sent_length is None:
        sent_length = max(len(sent) for sent in all_sents)

    output = []

    for story_set in story_sets:
        story_set_array = np.zeros((len(story_set), num_sents, sent_length))
        for story_num, story in enumerate(story_set):
            for sent_num, sent in enumerate(story):
                sent_pad = num_sents - len(story)
                word_pad = sent_length - len(sent)
                story_set_array[
                    story_num, sent_num + sent_pad, word_pad : len(sent) + word_pad
                ] = sent
        output.append(story_set_array)

    return output


def get_sent_hints(stories, hints):
    masks = np.zeros(stories.shape[:-1])
    for story_num, (story, story_hints) in enumerate(zip(stories, hints)):
        sent_pad = 0
        for sent in story:
            if np.any(sent):
                break
            sent_pad += 1
        for hint in story_hints:
            masks[story_num, hint + sent_pad] = 1
    return masks


# @memory.cache
def get_babi_data(
    task_subset=None,
    use_10k=False,
    forced_story_length=None,
    forced_question_length=None,
    forced_answer_length=None,
    forced_num_sents=None,
    forced_sent_length=None,
):
    babi_path = _babi_10K_path if use_10k else _babi_1K_path
    train_tasks = load_tasks("train", babi_path, task_subset)
    test_tasks = load_tasks("test", babi_path, task_subset)
    data = get_train_val_test(train_tasks, test_tasks, task_subset)

    data.word_to_vec, data.word_to_id, data.embedding_matrix = get_babi_embeddings(
        use_10k
    )
    data.id_to_word = {id: word for word, id in data.word_to_id.items()}

    # Convert lists of texts to lists of lists of word ids.
    keys = [
        key for key in data.keys() if ("X_" in key or "y_" in key) and "hint" not in key
    ]
    for key in keys:
        data[key] = texts_to_ids(data[key], data.word_to_id)

    data.X_train_stories, data.X_val_stories, data.X_test_stories = align_padding(
        data.X_train_stories,
        data.X_val_stories,
        data.X_test_stories,
        forced_length=forced_story_length,
    )
    data.X_train_questions, data.X_val_questions, data.X_test_questions = align_padding(
        data.X_train_questions,
        data.X_val_questions,
        data.X_test_questions,
        forced_length=forced_question_length,
    )
    data.y_train, data.y_val, data.y_test = align_padding(
        data.y_train, data.y_val, data.y_test, forced_length=forced_answer_length
    )

    data.X_train_decoder = get_time_shifted(data.y_train)
    data.X_val_decoder = get_time_shifted(data.y_val)
    data.X_test_decoder = get_time_shifted(data.y_test)

    # Add sentence masks.
    end_of_sentence_symbol = data.word_to_id["."]
    data.X_train_story_masks = get_sentence_masks(
        data.X_train_stories, end_of_sentence_symbol
    )
    data.X_val_story_masks = get_sentence_masks(
        data.X_val_stories, end_of_sentence_symbol
    )
    data.X_test_story_masks = get_sentence_masks(
        data.X_test_stories, end_of_sentence_symbol
    )

    # Expand attention hints.
    data.X_train_hints = get_hint_masks(
        data.X_train_stories, data.train_hints, end_of_sentence_symbol
    )
    data.X_val_hints = get_hint_masks(
        data.X_val_stories, data.val_hints, end_of_sentence_symbol
    )
    data.X_test_hints = get_hint_masks(
        data.X_test_stories, data.test_hints, end_of_sentence_symbol
    )

    # Add sentence versions of stories.
    data.X_train_story_sents = get_story_sents(
        data.X_train_stories, end_of_sentence_symbol
    )
    data.X_val_story_sents = get_story_sents(data.X_val_stories, end_of_sentence_symbol)
    data.X_test_story_sents = get_story_sents(
        data.X_test_stories, end_of_sentence_symbol
    )

    # Pad them.
    (
        data.X_train_story_sents,
        data.X_val_story_sents,
        data.X_test_story_sents,
    ) = align_story_sents(
        data.X_train_story_sents,
        data.X_val_story_sents,
        data.X_test_story_sents,
        num_sents=forced_num_sents,
        sent_length=forced_sent_length,
    )

    data.y_train_att = get_sent_hints(data.X_train_story_sents, data.train_hints)
    data.y_val_att = get_sent_hints(data.X_val_story_sents, data.val_hints)
    data.y_test_att = get_sent_hints(data.X_test_story_sents, data.test_hints)

    return data
