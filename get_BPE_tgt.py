import re, collections
from matplotlib import pyplot as plt

def get_vocab(filename):
    """
    :param filename: 文件地址，需要切好词，空格隔开
    :return: dic ：{ 'str str .. </w>' : 频率int  }
    """
    vocab = collections.defaultdict(int)
    class_num_list = []
    with open(filename, 'r', encoding='utf-8') as fhand:
        for line in fhand:
            words = line.strip().split()
            for word in words:
                vocab[' '.join(list(word)) + ' </w>'] += 1
    return vocab, class_num_list


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():  # 遍历每个单词中的 pair
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs


def merge_vocab(pair, v_in):
    """
    :param pair:
    :param v_in: 将 v_in 中的所有 pair 全部拼接
    :return:
    """
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out


def get_tokens(vocab):
    """ 得到 字符频率字典 """
    tokens = collections.defaultdict(int)
    for word, freq in vocab.items():
        word_tokens = word.split()  # [ 'str', 'str', ... ]
        for token in word_tokens:  # 遍历 str，放入tokens字典中
            tokens[token] += freq
    return tokens


def save_keys(tokens, path):
    with open(path, "w", encoding='utf-8') as f:
        for i in tokens.keys():
            f.write(i + "\n")


def save_items(tokens, path):
    with open(path, "w", encoding='utf-8') as f:
        for i, j in tokens:
            f.write(i + ' ' + j + "\n")


if __name__ == "__main__":
    """
    # 目的是压缩词表，所以设置目标 size
    """
    original_path = "../DC_train.src.txt"  # 目标文本 使用 src 模拟
    output_tgt_path = "BPE_DC_train.tgt.txt"  # 处理后文本
    output_token_path = "DC_vocab.txt"  # 处理后的词表
    # 两个东西共同设置迭代次数
    min_vocab_size = -1  # 默认可以设置为 -1
    merge_num = 400  # 默认可以设置为 999999999999

    vocab = get_vocab(original_path)  # word频率表
    # if len(vocab) > min_vocab_size:
    #     print("The word size is {}, bigger than min_vocab_size {},"
    #           " Please try bigger min_vocab_size !!!".format(len(vocab), min_vocab_size))
    #     exit()

    print('==========')
    print('Tokens Before BPE')
    tokens = get_tokens(vocab)  # token频率表 （统计用，训练用不到）理想情况下先增大再减小，再增大，再减小到和vocab大小一致。
    print('Tokens: {}'.format(tokens))
    print('Number of tokens: {}'.format(len(tokens)))
    print('==========')
    token_num = []
    for i in range(merge_num):
        if len(tokens) <= min_vocab_size:
            break
        pairs = get_stats(vocab)  # token pair频率表 （训练用，不会保存）
        if not pairs:  # 全为整词（len(symbols)-1==0）
            break
        best = max(pairs, key=pairs.get)  # 找最高频 pair
        # 更新 vocab 和 token 频率表
        vocab = merge_vocab(best, vocab)  # 融合 word词频表 中pair
        # print('Best pair: {}'.format(best))
        tokens = get_tokens(vocab)  # 重新得到 token词频表
        # print('Tokens: {}'.format(tokens))
        # print('Number of tokens: {}'.format(len(tokens)))
        # print('==========')
        token_num.append(len(tokens))

    print("Finally, Number of tokens: {}".format(len(tokens)))


    plt.plot(range(len(token_num)), token_num)
    plt.show()

    save_keys(tokens, output_token_path)
    save_keys(vocab, output_tgt_path)
