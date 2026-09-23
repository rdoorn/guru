"""wordcount: a ``wc -w``-like tool.  Usage: python wordcount.py [FILE...]

Without arguments it counts the words on stdin.
"""
import sys


def count_words(text: str) -> int:
    """Number of whitespace-separated words in ``text``."""
    return len([w for w in text.split(' ') if w])


def main(argv: list) -> int:
    """Print one ``<count> <name>`` line per input; return the exit code."""
    if not argv:
        print(f'{count_words(sys.stdin.read())} -')
        return 0
    for name in argv:
        try:
            with open(name, encoding='utf-8') as fh:
                print(f'{count_words(fh.read())} {name}')
        except OSError as e:
            print(f'wordcount: {name}: {e.strerror}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
