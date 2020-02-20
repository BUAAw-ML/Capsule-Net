import data_helpers
#
X_trn, Y_trn, Y_trn_o, X_tst, Y_tst, Y_tst_o, vocabulary, vocabulary_inv = data_helpers.load_data('eurlex_raw_text.p',
                                                                                max_length=500,
                                                                                vocab_size=300)
#
# print(len(vocabulary),len(vocabulary_inv))
print(len(X_trn),len(X_tst))
#
#
#
#
X_trn, Y_trn, Y_trn_o, X_tst, Y_tst, Y_tst_o, vocabulary, vocabulary_inv = data_helpers.load_data('API_classify_data(Programweb).p',
                                                                                max_length=500,
                                                                                vocab_size=300)
# print(len(vocabulary),len(vocabulary_inv))
print(len(X_trn),len(X_tst))
#
# import numpy as np
#
# def transformLabels(labels):
#     label_index = list(set([l for _ in labels for l in _]))
#     label_index.sort()
#
#     variable_num_classes = len(label_index)
#     target = []
#     for _ in labels:
#         tmp = np.zeros([variable_num_classes], dtype=np.float32)
#         tmp[[label_index.index(l) for l in _]] = 1
#         target.append(tmp)
#     target = np.array(target)
#     return label_index, target
#
# labels = [['1','3'],['1','3','8']]
# for i in transformLabels((labels)):
#     print(i)
