"""A checkpoint-defined vocabulary with one token per Unicode code point."""

from collections.abc import Iterable
from operator import index


class CharacterTokenizer:
    eos_token_id = 0
    bos_token_id = 1

    def __init__(self, chars: Iterable[str]):
        self._chars = tuple(chars)
        if not self._chars or any(
            not isinstance(char, str) or len(char) != 1 for char in self._chars
        ):
            raise ValueError("vocabulary must contain individual Unicode characters")
        if len(set(self._chars)) != len(self._chars):
            raise ValueError("vocabulary must not contain duplicate characters")
        self._ids = {char: token_id for token_id, char in enumerate(self._chars, 2)}

    @property
    def chars(self) -> tuple[str, ...]:
        return self._chars

    @property
    def vocab_size(self) -> int:
        return len(self._chars) + 2

    def encode(self, text: str) -> list[int]:
        try:
            return [self._ids[char] for char in text]
        except KeyError as error:
            raise ValueError(f"character {error.args[0]!r} is outside the vocabulary") from error

    def decode(self, token_ids: Iterable[int]) -> str:
        chars = []
        for value in token_ids:
            try:
                token_id = index(value)
            except TypeError as error:
                raise ValueError("token IDs must be integers") from error
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID {token_id} is outside the vocabulary")
            if token_id >= 2:
                chars.append(self._chars[token_id - 2])
        return "".join(chars)
