# import collections
# import re
# import string
#
#
# def tokenize_chinese_chars(text):
#     """
#     :param text: input text, unicode string
#     :return:
#         tokenized text, list
#     """
#
#     def _is_chinese_char(cp):
#         """Checks whether CP is the codepoint of a CJK character."""
#         # This defines a "chinese character" as anything in the CJK Unicode block:
#         #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
#         #
#         # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
#         # despite its name. The modern Korean Hangul alphabet is a different block,
#         # as is Japanese Hiragana and Katakana. Those alphabets are used to write
#         # space-separated words, so they are not treated specially and handled
#         # like the all of the other languages.
#         if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
#             (cp >= 0x3400 and cp <= 0x4DBF) or  #
#             (cp >= 0x20000 and cp <= 0x2A6DF) or  #
#             (cp >= 0x2A700 and cp <= 0x2B73F) or  #
#             (cp >= 0x2B740 and cp <= 0x2B81F) or  #
#             (cp >= 0x2B820 and cp <= 0x2CEAF) or
#             (cp >= 0xF900 and cp <= 0xFAFF) or  #
#             (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
#             return True
#         return False
#
#     output = []
#     buff = ""
#     for char in text:
#         cp = ord(char)
#         if _is_chinese_char(cp) or char == "=":
#             if buff != "":
#                 output.append(buff)
#                 buff = ""
#             output.append(char)
#         else:
#             buff += char
#
#     if buff != "":
#         output.append(buff)
#
#     return output
#
#
# def normalize_answer(s):
#     """
#     经过英文转小写、去除标点符号、增加空格来让答案和预测结果格式一致
#     Lower text and remove punctuation, articles and extra whitespace.
#     """
#
#     def remove_articles(text):
#         # re.U(UNICODE): 根据unicode字符集解析\w \W \b \B \s \S \d \D
#         # \b匹配单词和空格之间的字符
#         regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
#         return re.sub(regex, " ", text)
#
#     def white_space_fix(text):
#         return " ".join(text.split())
#
#     def remove_punc(text):
#         # string.punctuation将给出所有的标点集
#         exclude = set(string.punctuation)
#         return "".join(ch for ch in text if ch not in exclude)
#
#     def lower(text):
#         return text.lower()
#
#     return white_space_fix(remove_articles(remove_punc(lower(s))))
# # def _normalize(in_str):
# #     """
# #     normalize the input unicode string
# #     """
# #     in_str = in_str.lower()
# #     sp_char = [
# #         u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
# #         u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
# #         u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
# #     ]
# #     out_segs = []
# #     for char in in_str:
# #         if char in sp_char:
# #             continue
# #         else:
# #             out_segs.append(char)
# #     return ''.join(out_segs)
#
# def get_tokens(s):
#     # 把答案字符串变成单个token组成的list
#     if not s:
#         return []
#     return normalize_answer(s).split()
#
#
# def compute_exact(a_gold, a_pred):
#     """
#     Args:
#         a_gold:一个标准答案字符串
#         a_pred:一个预测答案字符串
#
#     Returns:预测答案是否和标准答案完全一致
#     """
#     return int(normalize_answer(a_gold) == normalize_answer(a_pred))
#
#
# def compute_f1(a_gold, a_pred):
#     """
#
#     Args:
#         a_gold: 一个标准答案字符串
#         a_pred: 一个预测答案字符串
#
#     Returns:由每个字计算的f1值
#
#     """
#     gold_toks = get_tokens(' '.join(tokenize_chinese_chars(a_gold)))
#     pred_toks = get_tokens(' '.join(tokenize_chinese_chars(a_pred)))
#     # 求交集， 例gold_toks有5个a,pred_toks有4个a，这样common有4个a
#     common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
#     # 计算元素总数，common.values()列表，每个元素是对应的数量，和common.keys()一一对应
#     num_same = sum(common.values())
#     if len(gold_toks) == 0 or len(pred_toks) == 0:
#         # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
#         return int(gold_toks == pred_toks)
#     if num_same == 0:
#         return 0
#     precision = 1.0 * num_same / len(pred_toks)
#     recall = 1.0 * num_same / len(gold_toks)
#     f1 = (2 * precision * recall) / (precision + recall)
#     return f1
#
#
# def get_raw_scores(examples, preds):
#     """
#     计算原始分数
#     Computes the exact and f1 scores from the examples and the model predictions
#     """
#     exact_scores = {}
#     f1_scores = {}
#
#     for example in examples:
#         qas_id = example['id']
#
#         if qas_id not in preds:
#             print(f"Missing prediction for {qas_id}")
#             continue
#
#         prediction = preds[qas_id]
#         if example['is_impossible'] and prediction == 'no answer':
#             exact_scores[qas_id] = 1
#             f1_scores[qas_id] = 1
#         elif any([not example['is_impossible'] and prediction == 'no answer', not example['is_impossible'] and prediction != 'no answer']):
#             exact_scores[qas_id] = 0
#             f1_scores[qas_id] = 0
#         else:
#             if len(example['answers']) == 0:
#                 raise ValueError('example answer should correspond to answer_starts')
#
#             exact_scores[qas_id] = compute_exact(example['answers'][0], prediction)
#             f1_scores[qas_id] = compute_f1(example['answers'][0], prediction)
#
#     return exact_scores, f1_scores
#
#
# def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
#     """
#     无答案阈值
#     Args:
#         scores:分数（exact分数或f1分数）
#         na_probs:{'id': 无答案概率}
#         qid_to_has_ans:{qid: True/False} qid有无答案的字典
#         na_prob_thresh:无答案阈值（默认为1.0）
#
#     Returns:经过修改的新分数，如果id的无答案概率大于阈值，那么预测有答案的分数为0.0，预测为无答案的分数为1.0。
#
#     """
#     new_scores = {}
#     for qid, s in scores.items():
#         pred_na = na_probs[qid] > na_prob_thresh
#         if pred_na:
#             new_scores[qid] = float(not qid_to_has_ans[qid])
#         else:
#             new_scores[qid] = s
#     return new_scores
#
#
# def make_eval_dict(exact_scores, f1_scores, qid_list=None):
#     """
#
#     Args:
#         exact_scores:exact分数
#         f1_scores:f1分数
#         qid_list:qid列表
#
#     Returns:字典：{'exact': exact总平均分, 'f1': f1总平均分, 'total': qid总个数}
#
#     """
#     if not qid_list:
#         total = len(exact_scores)
#         return collections.OrderedDict(
#             [
#                 ("exact", 100.0 * sum(exact_scores.values()) / total),
#                 ("f1", 100.0 * sum(f1_scores.values()) / total),
#                 ("total", total),
#             ]
#         )
#     else:
#         total = len(qid_list)
#         return collections.OrderedDict(
#             [
#                 ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
#                 ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
#                 ("total", total),
#             ]
#         )
#
#
# def merge_eval(main_eval, new_eval, prefix):
#     for k in new_eval:
#         main_eval[f"{prefix}_{k}"] = new_eval[k]
#
#
# def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
#     """
#     找到最佳无答案阈值
#     Args:
#         preds:
#         scores:
#         na_probs:字典，key为id，value为无答案概率
#         qid_to_has_ans:
#
#     Returns:能得到最高分的无答案阈值
#
#     """
#     num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k])
#     cur_score = num_no_ans
#     best_score = cur_score
#     best_thresh = 0.0
#     qid_list = sorted(na_probs, key=lambda k: na_probs[k])
#
#     for _, qid in enumerate(qid_list):
#         if qid not in scores:
#             continue
#         if qid_to_has_ans[qid]:
#             diff = scores[qid]
#         else:
#             if preds[qid] != 'no answer':
#                 diff = -1
#             else:
#                 diff = 0
#         cur_score += diff
#         if cur_score > best_score:
#             best_score = cur_score
#             best_thresh = na_probs[qid]
#
#     return 100.0 * best_score / len(scores), best_thresh
#
#
# def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
#
#     best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
#     best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)
#
#     main_eval["best_exact"] = best_exact
#     main_eval["best_exact_thresh"] = exact_thresh
#     main_eval["best_f1"] = best_f1
#     main_eval["best_f1_thresh"] = f1_thresh
#
#
# def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
#
#     qas_id_to_has_answer = {}
#     for example in examples:
#         if example['is_impossible']:
#             qas_id_to_has_answer[example['id']] = False
#         else:
#             qas_id_to_has_answer[example['id']] = True
#     has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
#     no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]
#
#     if no_answer_probs is None:
#         no_answer_probs = {k: 0.0 for k in preds}
#
#     exact, f1 = get_raw_scores(examples, preds)
#
#     exact_threshold = apply_no_ans_threshold(
#         exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
#     )
#     f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)
#
#     evaluation = make_eval_dict(exact_threshold, f1_threshold)
#
#     if has_answer_qids:
#         has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
#         merge_eval(evaluation, has_ans_eval, "HasAns")
#
#     if no_answer_qids:
#         no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
#         merge_eval(evaluation, no_ans_eval, "NoAns")
#
#     if no_answer_probs:
#         find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)
#
#     return evaluation
#



def _tokenize_chinese_chars(text):
    """
    :param text: input text, unicode string
    :return:
        tokenized text, list
    """

    def _is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    output = []
    buff = ""
    for char in text:
        cp = ord(char)
        if _is_chinese_char(cp) or char == "=":
            if buff != "":
                output.append(buff)
                buff = ""
            output.append(char)
        else:
            buff += char

    if buff != "":
        output.append(buff)

    return output


def _normalize(in_str):
    """
    normalize the input unicode string
    """
    in_str = in_str.lower()
    sp_char = [
        u':', u'_', u'`', u'，', u'。', u'：', u'？', u'！', u'(', u')',
        u'“', u'”', u'；', u'’', u'《', u'》', u'……', u'·', u'、', u',',
        u'「', u'」', u'（', u'）', u'－', u'～', u'『', u'』', '|'
    ]
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


def find_lcs(s1, s2):
    """find the longest common subsequence between s1 ans s2"""
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    max_len = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > max_len:
                    max_len = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - max_len:p], max_len


def f1_em_metric(ref_ans, pred_ans, verbose=False):
    """
    ref_ans: reference answers, dict
    pred_ans: predicted answer, dict
    return:
        f1_score: averaged F1 score
        em_score: averaged EM score
        total_count: number of samples in the reference dataset
        skip_count: number of samples skipped in the calculation due to unknown errors
    """
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    for sample in ref_ans:
        query_id = sample['id']
        total_count += 1
        para = sample['context']
        query_text = sample['question']
        title = sample['title']
        answers = sample['answers']
        is_impossible = sample['is_impossible']
        try:
            prediction = pred_ans[str(query_id)]
        except:
            skip_count += 1
            if verbose:
                print("para: {}".format(para))
                print("query: {}".format(query_text))
                print("ref: {}".format('#'.join(answers)))
                print("Skipped")
                print('----------------------------')
            continue
        if is_impossible:
            if prediction.lower() == 'no answer':
                _f1 = 1.0
                _em = 1.0
            else:
                _f1 = 0.0
                _em = 0.0
        else:
            _f1 = calc_f1_score(answers, prediction)
            _em = calc_em_score(answers, prediction)
        f1 += _f1
        em += _em
        if verbose:
            print("para: {}".format(para))
            print("query: {}".format(query_text))
            print("title: {}".format(title))
            print("ref: {}".format('#'.join(answers)))
            print("cand: {}".format(prediction))
            print("score: {}".format(_f1))
            print('----------------------------')

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return f1_score, em_score, total_count, skip_count


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = _tokenize_chinese_chars(_normalize(ans))
        prediction_segs = _tokenize_chinese_chars(_normalize(prediction))
        # if args.debug:
        #     print(json.dumps(ans_segs, ensure_ascii=False))
        #     print(json.dumps(prediction_segs, ensure_ascii=False))
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        prec = 1.0 * lcs_len / len(prediction_segs)
        rec = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * prec * rec) / (prec + rec)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = _normalize(ans)
        prediction_ = _normalize(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em



