import scipy.stats as stats

#### COMPUTE CORRELATION WITH ACCURACY RESULTS AS ANNOTATED BY 5 WORKERS ####
#Order of the students: WordRNN10, WordRNN07, WordRNN05, GoogleLM, AttentionAC, NoAttentionAC, SkipConnections, MLESeqGAN, SS, SeqGAN, RankGAN, LeakGAN
bleu_top10 = [ 0.2068, 0.3689, 0.4619, 0.0907, 0.3735, 0.5967, 0.5102, 0.1463, 0.1510, 0.1563, 0.1240, 0.1611 ]

rouge_top10 = [ 0.4234, 0.5200, 0.5670, 0.4847, 0.5245, 0.6592, 0.5824, 0.4217, 0.3964, 0.4292, 0.4103, 0.3366 ]

######### COMPUTE CORRELATION WITH ACCURACY RESULTS AS ANNOTATED BY THE MAJORITY VOTE ############
#Order of the students: WordRNN10, WordRNN07, WordRNN05, GoogleLM, AttentionAC, NoAttentionAC, SkipConnections, MLESeqGAN, SS, SeqGAN, RankGAN, LeakGAN
bleu_all = [0.6618, 0.8927, 0.9465, 0.4512, 0.8747, 0.9711, 0.9311, 0.5384, 0.5380, 0.5597, 0.5083, 0.4712]

rouge_all = [0.4962, 0.5744, 0.6110, 0.4852, 0.5827, 0.6924, 0.6339, 0.5027, 0.4945, 0.5092, 0.4884, 0.4286]

######## Kendall-tau ##########

bleu_tau, bleu_p_value = stats.kendalltau(bleu_top10, bleu_all)
print "BLEU Kendall-tau:", bleu_tau, "P-value:", bleu_p_value

rouge_tau, rouge_p_value = stats.kendalltau(rouge_top10, rouge_all)
print "ROUGE Kendall-tau:", rouge_tau, "P-value:", rouge_p_value


########### Spearman ##########
print 30 * "*"
bleu_tau, bleu_p_value = stats.spearmanr(bleu_top10, bleu_all)
print "BLEU Spearman:", bleu_tau, "P-value:", bleu_p_value

rouge_tau, rouge_p_value = stats.spearmanr(rouge_top10, rouge_all)
print "ROUGE Spearman:", rouge_tau, "P-value:", rouge_p_value


####### Pearson correlation #########
print 30 * "*"
bleu_tau, bleu_p_value = stats.pearsonr(bleu_top10, bleu_all)
print "BLEU Pearson:", bleu_tau, "P-value:", bleu_p_value

cnn_tau, cnn_p_value = stats.pearsonr(rouge_top10, rouge_all)
print "ROUGE Pearson:", cnn_tau, "P-value:", cnn_p_value

