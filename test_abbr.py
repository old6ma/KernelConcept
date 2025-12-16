"""
Abbreviation generation module for phrase matching.

This module provides functions to generate abbreviations from phrases using various
strategies including prefix generation, syllable-based splitting, and consonant skeleton
extraction.
"""

import re
import itertools
from functools import lru_cache
from typing import Set, List, Dict, Tuple
import nltk
from nltk.corpus import stopwords
from srctoolkit.delimiter import Delimiter
from tqdm import tqdm
from collections import defaultdict
from kernelTokenizer.kernel_delimiter import kernel_delimiter
import time
from NameHandler import NameHandler
# -------------------------
# Constants
# -------------------------
# Download stopwords if not already available
try:
    STOP_WORDS = frozenset(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = frozenset(stopwords.words('english'))

VOWELS = frozenset("aeiou")
NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z]+")
TIME_OUT=1

handler=NameHandler.get_inst()

# =========================
# 1. phrase tokenizer
# =========================

def tokenize(phrase: str, split_camel_case=False) -> List[str]:
    """
    Tokenize a phrase into lowercase alphabetic tokens.

    Args:
        phrase: Input phrase to tokenize

    Returns:
        List of lowercase alphabetic tokens
    """
    phrase = phrase.strip()
    if not split_camel_case:
        return phrase.lower().split()

    tokens = []
    for word in phrase.split():
        for part in kernel_delimiter(word).split():
            for token in NON_ALPHA_PATTERN.split(part):
                if token:
                    tokens.append(token.lower())

    return tokens

# =========================
# 2. Prefix family generation
# =========================
def prefix_family(word: str, max_len: int = 8) -> Tuple[str, ...]:
    """Generate all prefixes of a word up to max_len. Cached for performance."""
    return tuple(word[:i] for i in range(1, min(len(word), max_len) + 1))

# # =========================
# # 3. Syllable-like prefix
# # =========================
# def syllable_like_prefix(word: str, max_len: int = 8) -> str:
#     """
#     Extract a syllable-like prefix from a word.

#     Returns prefix up to the second vowel or max_len, whichever is shorter.
#     Returns None if word has no vowels.
#     """
#     if not word:
#         return None

#     first_vowel_idx = next((i for i, ch in enumerate(word) if ch in VOWELS), -1)
#     if first_vowel_idx == -1:
#         return None

#     second_vowel_idx = next(
#         (j for j in range(first_vowel_idx + 1, len(word)) if word[j] in VOWELS),
#         -1
#     )

#     if second_vowel_idx == -1:
#         return word[:max_len]

#     return word[:min(second_vowel_idx, max_len)]

# =========================
# 4. Consonant skeleton + subsequence variants
# =========================
def consonant_subsequence_variants(word: str, max_len: int = 8) -> Tuple[str, ...]:
    """
    Generate consonant skeleton subsequences.

    All generated sequences start with the original word's first letter.
    Uses DFS to generate subsequences with length between 2 and max_len.

    Args:
        word: Input word to process
        max_len: Maximum length of generated variants

    Returns:
        Sorted list of consonant skeleton variants
    """
    if not word:
        return []

    word_lower = word.lower()
    first_letter = word_lower[0]
    consonants = [c for i, c in enumerate(word_lower) if c not in VOWELS or word_lower[i-1] in {'/'}]

    if not consonants:
        return []

    variants = set()
    num_consonants = len(consonants)

    start=time.time()
    def dfs(idx: int, path: List[str]) -> None:
        """Generate subsequences using depth-first search."""
        path_len = len(path)
        end=time.time()
        if 1 < path_len <= max_len:
            variants.add("".join(path))

        if idx >= num_consonants or path_len >= max_len:
            return
        
        if end-start>TIME_OUT: #超时强制截断，避免指数爆炸堵死
            return 
        
        # Include current consonant
        dfs(idx + 1, path + [consonants[idx]])
        # Skip current consonant
        dfs(idx + 1, path)

    # Start DFS from second consonant with first letter fixed
    dfs(1, [first_letter])

    # Add full consonant skeleton truncated to max_len
    variants.add("".join(consonants[:max_len]))

    # Special case: if first letter is vowel and second is consonant, add second consonant. Eg., "extensible" -> "x"
    if len(word_lower) > 1 and word_lower[0] not in consonants and word_lower[1] in consonants:
        variants.add(word_lower[1])

    return tuple(sorted(variants))


# =========================
# 5. Candidates for each word
# =========================
@lru_cache(maxsize=4096)
def word_candidates(word: str, max_len: int = 8) -> Tuple[str, ...]:
    """
    Generate all abbreviation candidates for a single word.

    Combines prefixes, syllable-based prefixes, and consonant skeleton variants.
    Stop words return simplified candidates. Cached for performance.

    Args:
        word: Word to generate candidates for
        max_len: Maximum length for candidates

    Returns:
        Tuple of abbreviation candidates (sorted by length)
    """
    # Limit max_len to 70% of word length
    effective_max_len = min(max_len, int(len(word) * 0.75))

    if word in STOP_WORDS:
        candidates = {'', word[0], word}
        if word in {"and", "or", "for", "to"}:
            candidates.add('/')
        if word == "to":
            candidates.add('2')
        if word == "for":
            candidates.add('4')
        return tuple(sorted(candidates, key=len))

    candidates = set()

    # Add prefix family
    candidates.update(prefix_family(word, effective_max_len))

    # # Add syllable-like prefix
    # syllable_prefix = syllable_like_prefix(word, effective_max_len)
    # if syllable_prefix:
    #     candidates.add(syllable_prefix)

    # Add consonant skeleton subsequences
    candidates.update(consonant_subsequence_variants(word, effective_max_len))

    # Only truncate candidates that exceed max_len (optimization)
    if effective_max_len < max_len:
        candidates = {c[:max_len] if len(c) > max_len else c for c in candidates}

    return tuple(sorted(candidates, key=len))

# =========================
# 6. Combine tokens into multi-word abbreviations
# =========================
def combine_word_candidates(
    word_candidate_dict,
    max_len: int = 8
) -> List[str]:
    """
    Generate multi-word abbreviations by combining candidates from each word.

    Generates combinations using:
    - All abbreviated parts
    - First word full + rest abbreviated
    - Abbreviated + last word full
    - First and last words full + middle abbreviated

    Args:
        word_candidate_dict: Dict or list of (word, candidates) tuples preserving order
        max_len: Maximum length for combined abbreviations

    Returns:
        Sorted list of valid abbreviation combinations
    """
    results = set()

    # Handle both dict and list input
    if isinstance(word_candidate_dict, dict):
        words, candidate_lists = zip(*word_candidate_dict.items())
    else:
        words, candidate_lists = zip(*word_candidate_dict)
    first_word = words[0]
    last_word = words[-1]
    is_multi_word = len(words) > 1
    num_words = len(words)

    # Detect consecutive words with same first letter (e.g., "SQL Standard Scalable" -> "s3")
    consecutive_pattern = None
    consecutive_start_idx = -1
    consecutive_end_idx = -1

    if num_words >= 3:
        # Build acronym from first letters
        first_letters = [word[0].lower() for word in words]

        # Find longest sequence of identical consecutive letters (minimum 3)
        for i in range(num_words - 2):
            letter = first_letters[i]
            count = 1

            # Count consecutive occurrences
            for j in range(i + 1, num_words):
                if first_letters[j] == letter:
                    count += 1
                else:
                    break

            # If we found 3+ consecutive same letters, create pattern
            if count >= 3 and (consecutive_pattern is None or count > int(consecutive_pattern[1:])):
                consecutive_pattern = letter + str(count)
                consecutive_start_idx = i
                consecutive_end_idx = i + count
    start=time.time()
    for combination in itertools.product(*candidate_lists):
        # Calculate total length once for reuse
        total_len = sum(len(c) for c in combination)

        # All parts abbreviated
        if total_len <= max_len:
            results.add("".join(combination))

        # Add consecutive letter pattern abbreviation (e.g., "as3ap" for "ANSI SQL Standard Scalable and Portable")
        if consecutive_pattern and consecutive_start_idx >= 0:
            # Build: prefix + pattern + suffix
            prefix_parts = combination[:consecutive_start_idx]
            suffix_parts = combination[consecutive_end_idx:]
            new_combination = prefix_parts + (consecutive_pattern,) + suffix_parts

            new_total_len = sum(len(c) for c in new_combination)
            if new_total_len <= max_len:
                results.add("".join(new_combination))

        if is_multi_word:
            comb_first_len = len(combination[0])
            comb_last_len = len(combination[-1])

            # First word full + rest abbreviated
            new_len = total_len - comb_first_len + 1
            if new_len <= max_len:
                results.add(first_word + "".join(combination[1:]))

            # Abbreviated + last word full
            new_len = total_len - comb_last_len + 1
            if new_len <= max_len:
                results.add("".join(combination[:-1]) + last_word)

            # First and last full + middle abbreviated
            new_len = total_len - comb_first_len - comb_last_len + 2
            if new_len <= max_len:
                results.add(first_word + "".join(combination[1:-1]) + last_word)
        end=time.time()
        if end-start>TIME_OUT:
            break

    return sorted(results, key=lambda x: (len(x), x))

# =========================
# 7. Main abbreviation function
# =========================
def abbreviate(
    phrase: str,
    max_part_len: int = 8,
    max_abbr_len: int = 8
) -> Set[str]:
    """
    Generate all possible abbreviations for a phrase.

    Args:
        phrase: Input phrase to abbreviate
        max_part_len: Maximum length for individual word abbreviations
        max_abbr_len: Maximum length for combined abbreviations

    Returns:
        Set of valid abbreviations (excluding the original phrase)
    """
    phrase_stripped = phrase.strip()

    if len(phrase_stripped) <= 3:
        return set()

    def _abbreviate_tokens(tokens: Tuple[str]) -> Set[str]:
        """Helper to abbreviate based on token list."""
        if not tokens:
            return set()

        if len(tokens) == 1:
            abbreviations = word_candidates(tokens[0], max_part_len)
        else:
            # Use list to preserve order and duplicates
            candidate_groups = [
                (word, word_candidates(word, max_part_len))
                for word in tokens
            ]
            abbreviations = combine_word_candidates(candidate_groups, max_abbr_len)

        # Exclude the original phrase (normalized)
        res=[]
        for abbr in abbreviations:
            if len(re.sub(r'[^a-zA-Z0-9]', '', abbr.lower()))<len(abbr)/2:#特殊情况如__s_，符号数过多
                continue
            if abbr != phrase.lower():
                if handler.check_abbr_simple(phrase.lower(),abbr,True):
                    res.append(abbr)
        return res
        # return {abbr for abbr in abbreviations if abbr != phrase.lower()and handler.check_abbr(phrase.lower(),abbr,True)}

    abbreviations = set()
    tokens_without_splitting_camelcase = tuple(tokenize(phrase, split_camel_case=False))
    abbreviations.update(_abbreviate_tokens(tokens_without_splitting_camelcase))
    tokens_with_splitting_camelcase = tuple(tokenize(phrase, split_camel_case=True))
    if tokens_with_splitting_camelcase != tokens_without_splitting_camelcase:
        abbreviations.update(_abbreviate_tokens(tokens_with_splitting_camelcase))
    return abbreviations


def pair(corpus: Set[str], verbose: bool = False) -> List[Tuple[str, str]]:
    """
    Find phrase-abbreviation pairs within a corpus.

    For each phrase, generates abbreviations and checks if any match other phrases
    in the corpus (with spaces normalized).

    Args:
        corpus: Set of phrases to analyze
        verbose: Whether to print progress information

    Returns:
        List of (phrase, matching_abbreviation) tuples
    """
    corpus = set(corpus)

    # Build normalized corpus mapping: normalized -> [original items]
    normalized_to_original = {}
    for item in corpus:
        if len(item)>40 and '_' in item :
            continue
        normalized = re.sub(r'[/]', ' ', item)
        normalized = re.split(r'[\[\(<]', normalized)[0].strip()
        normalized = re.sub(r'-', ' ', normalized).lower()
        normalized = normalized.strip(".,/\\!@#$%^&*()-_=+`~\"';:<>?")
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized_to_original.setdefault(normalized, []).append(item)
        #特殊情况："High-Level Data Link Control"->HighLevel Data Link Control
        if '-' in item:
            normalized=item.replace('-','')
            normalized = re.split(r'[\[\(<]', normalized)[0].strip()
            normalized = normalized.strip(".,/\\!@#$%^&*()-_=+`~\"';:<>?")
            normalized = re.sub(r'\s+', ' ', normalized)
            normalized_to_original.setdefault(normalized, []).append(item)
    normalized_corpus = frozenset(normalized_to_original.keys())

    compact_to_original = {}
    for item in normalized_corpus:
        compact = item.replace(' ', '').lower()
        compact_to_original.setdefault(compact, []).extend(normalized_to_original[item])
    compact_corpus = frozenset(compact_to_original.keys())

    results = defaultdict(list)
    for phrase in tqdm(normalized_corpus):
        abbreviations = abbreviate(phrase, max_part_len=8, max_abbr_len=8)
        if verbose:
            print(f"Phrase: {normalized_to_original[phrase]} => Abbrs: {abbreviations}")

        # Find matches in normalized corpus (set intersection is faster than loop)
        matching_abbrs = abbreviations & compact_corpus

        # Collect all matches at once 
        for abbr in matching_abbrs:
            for full, abbr in itertools.product(normalized_to_original[phrase], compact_to_original[abbr]):
                if full != abbr:
                    results[full].append(abbr)
    return results


import random, json


def test(sample_num=100):
    nums = random.sample(range(0, 3701), sample_num)
    with open('./abbreviation_dict.json', 'r') as f:
        datas = json.load(f)
        sample_keys = [list(datas.keys())[i] for i in nums]
        ground_truth = []
        corpus = []
        for key in sample_keys:
            if '+' in datas[key]:
                corpus.append(key)
                for ans in datas[key].split('+'):
                    if len(ans)>50:
                        continue
                    ground_truth.append((ans.strip(), key.strip()))
                    corpus.append(ans.strip())
                continue

            if len(key)>50 or len(datas[key])>50 and '_' in datas[key]:
                continue
    
            ground_truth.append((datas[key].strip(), key.strip()))
            corpus.append(datas[key].strip())
            corpus.append(key.strip())

        match_res = pair(corpus)
    matched = []
    for long_name,abbr_res in match_res.items():
        for res in abbr_res:
            if (long_name,res) in ground_truth:
                matched.append((long_name,res))
    print(len(matched) / len(ground_truth))

    miss_res = []
    for gt in ground_truth:
        if gt not in matched:
            miss_res.append(gt)
    with open('./test_result_3000.json', 'w') as f:
        json.dump({"matched": matched, "missed": miss_res, "match_res":match_res,"corpus":corpus,"recall": len(matched) / len(ground_truth)}, f, indent=4,
                  ensure_ascii=False)


# =========================
# 8. Main execution
# =========================
if __name__ == "__main__":
    # print(pair([
    #         "Dual In-line Package",
    #         "dip"
    #     ]))
    # abbreviate("dpcsrx_rx_cntl__dpcs_rx_lane0_en_mask")
    test(3000)
    # abbreviate("Receive/Transmit")
    #
    # # Test corpus with common phrases and their abbreviations
    # test_corpus = {
    #     "statement",
    #     "stmt",
    #     "architecture",
    #     "arch",
    #     "performance",
    #     "perf",
    #     "management",
    #     "mgmt",
    #     "configuration",
    #     "config",
    #     "artificial neural network",
    #     "artificial intelligence",
    #     "hypertext markup language",
    #     "html",
    #     "asynchronous io",
    #     "async io",
    #     "aio",
    #     "asyncio",
    #     "ata packet interface",
    #     "adaptive quality of service architecture",
    #     "aquosa",
    #     "alsa system on chip",
    #     "binary-coded decimal",
    #     "grand unified bootloader",
    #     "grub",
    #     "massage",
    #     "page table",
    #     "pgtable",
    #     "kernel memory",
    #     "input output control",
    #     "recurrent neural network",
    #     "recurrent nn",
    #     "memory management",
    #     "mem management",
    #     "input/output map",
    #     "io map",
    #     "arm64 architecture",
    #     "arm64arch",
    #     "Receive/Transmit",
    #     "r/t",
    #     "Local Multipoint Distribution Service",
    #     "lmds",
    #     "Below or Equal",
    #     "be",
    #     "Turbo Editor Macro Language [Borland]",
    #     "teml",
    #     "Link/Local Management Interface",
    #     "lmi",
    #     "Network I/Os Per Second",
    #     "nips",
    #     "Fiber Channel/Enhanced Loop",
    #     "fc/el",
    #     "Lookup Table",
    #     "lut",
    #     "Compact Disk - Real Time Operating System",
    #     "cd-rtos",
    #     "Extended Memory Specification [LIM/AST]",
    #     "xms",
    #     "ANSI SQL Standard Scalable and Portable",
    #     "as3ap",
    #     "Indexed Sequential-Access Management/Method",
    #     "isam"
    # }
    #
    # # Run pair matching on test corpus
    # matches = pair(test_corpus, verbose=True)
    # print("\nFound matches:")
    # for phrase, match in matches:
    #     print(f"  {phrase!r} -> {match!r}")
