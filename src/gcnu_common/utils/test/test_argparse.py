
import argparse
import gcnu_common.utils.argparse

def test_add_remaining_to_populated_args():
    p = argparse.ArgumentParser()
    p.add_argument('pos1')
    p.add_argument('pos2')

    populated, remaining = p.parse_known_args('1 2 --key1=8 key2 9'.split())
    gcnu_common.utils.argparse.add_remaining_to_populated_args(
        populated=populated, remaining=remaining)

    assert(populated.pos1 == "1")
    assert(populated.pos2 == "2")
    assert(populated.key1 == "8")
    assert(populated.key2 == "9")

if __name__ == "__main__":
    test_add_remaining_to_populated_args()

