# from string import punctuation
# from string import whitespace
#
# def tokenize(text):
#     """
#     Tokenizes the input text into words by splitting on whitespace and punctuation.
#
#     Args:
#         text (str): The input text to be tokenized.
#     Returns:
#         list: A list of tokens (words).
#     """
#
#     tokens = []
#     current_token = []
#
#     for prev, char in zip([None] + list(text[:-1]), text):
#         if char in whitespace or char in punctuation:
#             if current_token:
#                 tokens.append(''.join(current_token))
#                 current_token = []
#             if char in punctuation and prev not in punctuation:
#                 tokens.append(char)
#             if char in whitespace and prev not in whitespace:
#                 tokens.append(char)
#         else:
#             current_token.append(char)
#
#     if current_token:
#         tokens.append(''.join(current_token))
#
#     return tokens
#
# print(tokenize("Hello, world! This is a   test ???"))