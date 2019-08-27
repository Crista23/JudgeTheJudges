import scipy.stats as stats

####  Word-overlap Results when considering the annotated D-test with human labels majority vote as fake and as ground-truth the most similar top10 sentences from D-train ####
## results computed with the scripts comput_metrics_human_labels_majority_vote.py and blue_evaluation_sent2vec_correctedbug_human_labels_majority_vote.py
bleu_evaluator = [ 0.204676, 0.361213, 0.454281, 0.094986, 0.365594, 0.565690, 0.451211, 0.134171, 0.141332, 0.154992, 0.109252, 0.138026 ]
meteor_evaluator = [ 0.248679, 0.310388, 0.353043, 0.285609, 0.289486, 0.365384, 0.326371, 0.235972, 0.230468,  0.244180, 0.241870, 0.174016 ]
rouge_evaluator = [ 0.416998, 0.512888, 0.558344, 0.492224,  0.534062, 0.656950, 0.577916, 0.422275,  0.393269, 0.438682,  0.402628, 0.319571 ]
cider_evaluator = [ 0.174060, 0.274607, 0.292306, 0.031977, 0.444226, 0.322985, 0.629278, 0.211741, 0.188780, 0.192950, 0.119744, 0.128580 ]

#### COMPUTE CORRELATION WITH ACCURACY RESULTS AS ANNOTATED BY 5 WORKERS ####
#Order of the students: WordRNN10, WordRNN07, WordRNN05, GoogleLM, AttentionAC, NoAttentionAC, SkipConnections, MLESeqGAN, SS, SeqGAN, RankGAN, LeakGAN
#human_accuracy = [0.54873164219, 0.3391188251, 0.26711409396, 0.681940700809, 0.323056300268, 0.387182910547, 0.246318607764, 0.762349799733, \
#			0.752673796791, 0.744966442953, 0.778225806452, 0.681392235609]
######### COMPUTE CORRELATION WITH ACCURACY RESULTS AS ANNOTATED BY THE MAJORITY VOTE ############
#Order of the students: WordRNN10, WordRNN07, WordRNN05, GoogleLM, AttentionAC, NoAttentionAC, SkipConnections, MLESeqGAN, SS, SeqGAN, RankGAN, LeakGAN
human_accuracy = [0.597315436242, 0.281879194631, 0.178082191781, 0.791666666667, 0.272108843537, 0.342281879195, 0.148648648649, 0.89932885906, \
		0.872483221477, 0.850340136054, 0.842465753425, 0.761904761905]
human_accuracy = [ 1 -x for x in human_accuracy ]

#### Kendall-tau ####
bleu_tau, bleu_p_value = stats.kendalltau(bleu_evaluator, human_accuracy)
print "BLEU Kendall-tau:", bleu_tau, "P-value:", bleu_p_value

rouge_tau, rouge_p_value = stats.kendalltau(rouge_evaluator, human_accuracy)
print "ROUGE Kendall-tau:", rouge_tau, "P-value:", rouge_p_value

meteor_tau, meteor_p_value = stats.kendalltau(meteor_evaluator, human_accuracy)
print "METEOR Kendall-tau:", meteor_tau, "P-value:", meteor_p_value

cider_tau, cider_p_value = stats.kendalltau(cider_evaluator, human_accuracy)
print "CIDER Kendall-tau:", cider_tau, "P-value:", cider_p_value


#### Spearman ####
bleu_tau, bleu_p_value = stats.spearmanr(bleu_evaluator, human_accuracy)
print "BLEU Spearman:", bleu_tau, "P-value:", bleu_p_value

rouge_tau, rouge_p_value = stats.spearmanr(rouge_evaluator, human_accuracy)
print "ROUGE Spearman:", rouge_tau, "P-value:", rouge_p_value

meteor_tau, meteor_p_value = stats.spearmanr(meteor_evaluator, human_accuracy)
print "METEOR Spearman:", meteor_tau, "P-value:", meteor_p_value

cider_tau, cider_p_value = stats.spearmanr(cider_evaluator, human_accuracy)
print "CIDER Spearman:", cider_tau, "P-value:", cider_p_value


##### Pearson ####
bleu_tau, bleu_p_value = stats.pearsonr(bleu_evaluator, human_accuracy)
print "BLEU Pearson:", bleu_tau, "P-value:", bleu_p_value

rouge_tau, rouge_p_value = stats.pearsonr(rouge_evaluator, human_accuracy)
print "ROUGE Pearson:", rouge_tau, "P-value:", rouge_p_value

meteor_tau, meteor_p_value = stats.pearsonr(meteor_evaluator, human_accuracy)
print "METEOR Pearson:", meteor_tau, "P-value:", meteor_p_value

cider_tau, cider_p_value = stats.pearsonr(cider_evaluator, human_accuracy)
print "CIDER Pearson:", cider_tau, "P-value:", cider_p_value