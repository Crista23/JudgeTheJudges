from nlgeval import compute_metrics
import pickle

### for each sentence included in the user study per generative model (150 examples/model) retrieve the top-n most similar sentences and write each to file ###
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_WordRNN10_annotated_examples.pickle", "rb") as handle:
#    student_WordRNN10_annotated_examples = pickle.load(handle)
#print ("len(student_WordRNN10_annotated_examples)", len(student_WordRNN10_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_WordRNN07_annotated_examples.pickle", "rb") as handle:
#    student_WordRNN07_annotated_examples = pickle.load(handle)
#print ("len(student_WordRNN07_annotated_examples)", len(student_WordRNN07_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_WordRNN05_annotated_examples.pickle", "rb") as handle:
#    student_WordRNN05_annotated_examples = pickle.load(handle)
#print ("len(student_WordRNN05_annotated_examples)", len(student_WordRNN05_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_GoogleLM_annotated_examples.pickle", "rb") as handle:
#    student_GoogleLM_annotated_examples = pickle.load(handle)
#print ("len(student_GoogleLM_annotated_examples)", len(student_GoogleLM_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_AttentionAC_annotated_examples.pickle", "rb") as handle:
#    student_AttentionAC_annotated_examples = pickle.load(handle)
#print ("len(student_AttentionAC_annotated_examples)", len(student_AttentionAC_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_NoAttentionAC_annotated_examples.pickle", "rb") as handle:
#    student_NoAttentionAC_annotated_examples = pickle.load(handle)
#print ("len(student_NoAttentionAC_annotated_examples)", len(student_NoAttentionAC_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_SkipConnections_annotated_examples.pickle", "rb") as handle:
#    student_SkipConnections_annotated_examples = pickle.load(handle)
#print ("len(student_SkipConnections_annotated_examples)", len(student_SkipConnections_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_MLESeqGAN_annotated_examples.pickle", "rb") as handle:
#    student_MLESeqGAN_annotated_examples = pickle.load(handle)
#print ("len(student_MLESeqGAN_annotated_examples)", len(student_MLESeqGAN_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_SS_annotated_examples.pickle", "rb") as handle:
#    student_SS_annotated_examples = pickle.load(handle)
#print ("len(student_SS_annotated_examples)", len(student_SS_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_SeqGAN_annotated_examples.pickle", "rb") as handle:
#    student_SeqGAN_annotated_examples = pickle.load(handle)
#print ("len(student_SeqGAN_annotated_examples)", len(student_SeqGAN_annotated_examples))

#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_RankGAN_annotated_examples.pickle", "rb") as handle:
#    student_RankGAN_annotated_examples = pickle.load(handle)
#print ("len(student_RankGAN_annotated_examples)", len(student_RankGAN_annotated_examples))

with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_LeakGAN_annotated_examples.pickle", "rb") as handle:
    student_LeakGAN_annotated_examples = pickle.load(handle)
print ("len(student_LeakGAN_annotated_examples)", len(student_LeakGAN_annotated_examples))

#generated_data_idx_test = student_WordRNN10_annotated_examples
#generated_data_idx_test = student_WordRNN07_annotated_examples
#generated_data_idx_test = student_WordRNN05_annotated_examples
#generated_data_idx_test = student_GoogleLM_annotated_examples
#generated_data_idx_test = student_AttentionAC_annotated_examples
#generated_data_idx_test = student_NoAttentionAC_annotated_examples
#generated_data_idx_test = student_SkipConnections_annotated_examples
#generated_data_idx_test = student_MLESeqGAN_annotated_examples
#generated_data_idx_test = student_SS_annotated_examples
#generated_data_idx_test = student_SeqGAN_annotated_examples
#generated_data_idx_test = student_RankGAN_annotated_examples
generated_data_idx_test = student_LeakGAN_annotated_examples

X_test = [ question_text for (question_text, true_label, human_label) in generated_data_idx_test if human_label == 0 ]
print len(X_test)

#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_wordrnn10.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_wordrnn07.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_wordrnn05.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_googlelm.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_attentionac.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_noattentionac.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_skipconnections.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_mleseqgan.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_ss.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_seqgan.txt", "w")
#z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_rankgan.txt", "w")
z = open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_leakgan.txt", "w")
for item in X_test:
	z.write(item + "\n")
z.close()


### 10 MOST SIMILAR REFERENCES USING THE HUMAN ANNOTATED EXAMPLES ###
metrics_dict = compute_metrics(#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_wordrnn10.txt', 
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_wordrnn07.txt', 
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_wordrnn05.txt', 
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_googlelm.txt',
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_attentionac.txt',
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_noattentionac.txt', 
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_skipconnections.txt',
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_mleseqgan.txt',
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_ss.txt', 
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_seqgan.txt',
					#hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_rankgan.txt',
					hypothesis='/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/hypothesis_files/hypothesis_leakgan.txt', 

					references=[


					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn10_most_similar/ref10.txt'
					

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn07_most_similar/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/word_rnn05_most_similar/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/google_lm/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/attention_ac/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/no_attention_ac/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/skip_connections/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/mle_seqgan/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/ss/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/seqgan/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/rankgan/ref10.txt'

					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref1.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref2.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref3.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref4.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref5.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref6.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref7.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref8.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref9.txt',
					'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references_correctedbug_human_labels_majority_vote/leakgan/ref10.txt'

					###################################

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/skip_connections/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/no_attention_ac/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/attention_ac/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn10_most_similar/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/leakgan/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/rankgan/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/google_lm/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/seqgan/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn07_most_similar/ref10.txt'

					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref1.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref2.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref3.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref4.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref5.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref6.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref7.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref8.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref9.txt',
					#'/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/nlp_metrics_eval/most_similar_references/word_rnn05_most_similar/ref10.txt'
					])
