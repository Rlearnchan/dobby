from typing import Dict, List

# by extract_words.process_and_save_word_counts with basic setting (random_state=42, test_size=0.2)

# sales_speed = {
#     "CH": {
#         0: 37, # chicago, single house, fast-selling 25%
#         1: 37,
#     },
#     "NY": {
#         0: 83,
#         1: 97,
#     },
#     "LA": {
#         0: 12,
#         1: 20,
#     },
# }

sales_speed: Dict[str, Dict[int, Dict[str, List[int]]]] = {
    "CH": {
        0: {"fast": [21, 28, 32, 35, 37, 40], "slow": [174, 129, 106, 89, 77, 69]}, # 5% to 30%
        1: {"fast": [20, 27, 32, 35, 37, 40], "slow": [152, 110, 93, 82, 74, 68]},
    },
    "NY": {
        0: {"fast": [54, 64, 70, 76, 83, 89], "slow": [272, 241, 220, 196, 171, 151]},
        1: {"fast": [55, 71, 81, 90, 97, 104], "slow": [280, 247, 235, 222, 212, 197]},
    },
    "LA": {
        0: {"fast": [2, 5, 7, 9, 12, 16], "slow": [128, 92, 77, 64, 56, 50]},
        1: {"fast": [4, 7, 10, 14, 20, 26], "slow": [153, 110, 92, 83, 75, 65]},
    }
}