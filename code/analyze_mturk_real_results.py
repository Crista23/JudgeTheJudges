from __future__ import division
import csv
from collections import defaultdict
import collections
from datetime import datetime
from dateutil.parser import parse
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle

def days_hours_minutes(td):
	return td.days, td.seconds//3600, (td.seconds//60)%60

def plot_annotation_time(mins_spent):
	# fixed bin size
	bins = np.arange(-100, 100, 5) # fixed bin size
	plt.xlim([min(mins_spent)-5, max(mins_spent)+5])

	plt.hist(mins_spent, bins=bins, alpha=0.5, histtype='bar', ec='black')
	plt.title('MTurk Worker Minutes')
	plt.xlabel('Minutes spent annotating (bin size = 5)')
	plt.ylabel('worker count')
	plt.savefig("worker_minutes_real_study_histogram.png", dpi=150)


## Determine which workers participated in the pilot study ##
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_eval_pilot/workers_pilot_study.pickle", "rb") as handle:
	worker_ids_pilot_study = pickle.load(handle)

print "*****Workers ids pilot study", worker_ids_pilot_study

## Determine which workers completed more than 1 HIT ##
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_duplicate.pickle", "rb") as handle:
	worker_ids_duplicate = pickle.load(handle)
duplicate_workers = [ worker_id for (worker_id, count) in worker_ids_duplicate ]
worker_ids_duplicate = duplicate_workers
print "******DUPLICATE WORKERS", worker_ids_duplicate

## Determine which workers did not answer all questions ##
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_incomplete_hits.pickle", "rb") as handle:
	worker_ids_incomplete = pickle.load(handle)
print "******Worker ids incomplete", worker_ids_incomplete

## Determine which workers failed to answer the gotcha question correctly ##
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_failed_gotcha.pickle", "rb") as handle:
	worker_ids_failed_gotcha = pickle.load(handle)
print "******Worker ids failed gotcha", worker_ids_failed_gotcha

workers_to_avoid = worker_ids_pilot_study.union(worker_ids_duplicate).union(worker_ids_incomplete).union(worker_ids_failed_gotcha)
print "Workers to avoid", workers_to_avoid
print "LEN(workers_to_avoid)", len(set(workers_to_avoid))


#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results.csv", "rb") as csvfile:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results_latest.csv", "rb") as csvfile:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results_3rd.csv", "rb") as csvfile:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results_4th.csv", "rb") as csvfile:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results_5th.csv", "rb") as csvfile:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results_6th.csv", "rb") as csvfile:
#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results_7th.csv", "rb") as csvfile:
with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/Batch_3205459_batch_results_8th.csv", "rb") as csvfile:
 	mturk_pilot_annotations = csv.DictReader(csvfile, delimiter=',')

 	hit_ids = defaultdict(int)
 	worker_ids = list()
 	worker_time = defaultdict(int)
 	banned_workers_failed_gotcha = list()
 	worker_answers = defaultdict(dict)
 	workers_incomplete_hits = list()
 	accuracy_per_student = defaultdict(dict)


 	tp_all = list()
 	tn_all = list()
 	fn_all = list()
 	fp_all = list()


 	student_WordRNN10 = defaultdict(int)
 	overall_annotations_per_student_WordRNN10 = defaultdict(lambda: defaultdict(int))

 	student_WordRNN07 = defaultdict(int)
 	overall_annotations_per_student_WordRNN07 = defaultdict(lambda: defaultdict(int))

 	student_WordRNN05 = defaultdict(int)
 	overall_annotations_per_student_WordRNN05 = defaultdict(lambda: defaultdict(int))

 	student_GoogleLM = defaultdict(int)
	overall_annotations_per_student_GoogleLM = defaultdict(lambda: defaultdict(int))

 	student_AttentionAC = defaultdict(int)
 	overall_annotations_per_student_AttentionAC = defaultdict(lambda: defaultdict(int))

 	student_NoAttentionAC = defaultdict(int)
	overall_annotations_per_student_NoAttentionAC = defaultdict(lambda: defaultdict(int)) 	

 	student_SkipConnections = defaultdict(int)
 	overall_annotations_per_student_SkipConnections = defaultdict(lambda: defaultdict(int))

 	student_MLESeqGAN = defaultdict(int)
 	overall_annotations_per_student_MLESeqGAN = defaultdict(lambda: defaultdict(int))

 	student_SS = defaultdict(int)
 	overall_annotations_per_student_SS = defaultdict(lambda: defaultdict(int))

 	student_SeqGAN = defaultdict(int)
 	overall_annotations_per_student_SeqGAN = defaultdict(lambda: defaultdict(int))

 	student_RankGAN = defaultdict(int)
 	overall_annotations_per_student_RankGAN = defaultdict(lambda: defaultdict(int))

 	student_LeakGAN = defaultdict(int)
 	overall_annotations_per_student_LeakGAN = defaultdict(lambda: defaultdict(int))

 	student_real = defaultdict(int)
 	overall_annotations_per_student_REAL = defaultdict(lambda: defaultdict(int))

 	student_fake = defaultdict(int)
	overall_annotations_per_student_FAKE = defaultdict(lambda: defaultdict(int))
 	
 	count_WordRNN10_all = 0
 	count_WordRNN07_all = 0
 	count_WordRNN05_all = 0
 	count_GoogleLM_all = 0
 	count_AttentionAC_all = 0
 	count_NoAttentionAC_all = 0
 	count_SkipConnections_all = 0
 	count_MLESeqGAN_all = 0
 	count_SS_all = 0
 	count_SeqGAN_all = 0
 	count_RankGAN_all = 0
 	count_LeakGAN_all = 0

 	count_real = 0
 	count_fake = 0
 	count_WordRNN10 = 0
 	count_WordRNN07 = 0
 	count_WordRNN05 = 0
 	count_GoogleLM = 0
 	count_AttentionAC = 0
 	count_NoAttentionAC = 0
 	count_SkipConnections = 0
 	count_MLESeqGAN = 0
 	count_SS = 0
 	count_SeqGAN = 0
 	count_RankGAN = 0
 	count_LeakGAN = 0

 	j = 0
	for row in mturk_pilot_annotations:
		#print "j=", j
		j += 1
		#print "ROW", row
		current_hit = row["HITId"]
		worker_id = row["WorkerId"]
		#print "Worker id", worker_id
		if (worker_id in workers_to_avoid):
			continue

		hit_ids[current_hit] += 1
		worker_ids.append(worker_id)

		#print "Original Start Time", row["AcceptTime"]
		#print "Original Start Time", row["SubmitTime"]

		start_time = row["AcceptTime"].split(" ") [0] + ", " + " ".join(row["AcceptTime"].split(" ")[1:4])
		end_time = row["SubmitTime"].split(" ") [0] + ", " + " ".join(row["SubmitTime"].split(" ")[1:4])
		
		worker_start_time = parse(start_time)
		worker_end_time = parse(end_time)

		#print "Worker start time", worker_start_time
		#print "Worker end time", worker_end_time

		minutes_spent_annotating = days_hours_minutes(worker_end_time - worker_start_time)[2]
		#print "TIME SPENT ANNOTATING (MINUTES)", minutes_spent_annotating
		worker_time[worker_id] = minutes_spent_annotating

		## Determine if the worker has correctly responded to the security question
		gotcha_loc = row["Input.gotcha_locs"]
		gotcha_question_answer = row["Answer.review_" + str(gotcha_loc) + "_judgment"]
		#print "GOTCHA ANSWER", gotcha_question_answer
		if gotcha_question_answer != "fake":
			banned_workers_failed_gotcha.append(worker_id)
			continue

		tp = 0
		tn = 0
		fn = 0
		fp = 0

		has_incomplete_answers = 0
		for i in range(0,21):
			# do not count the security question and its answer
			#print "i=", i
			if i == int(gotcha_loc):
				continue

			question_answer_worker = row["Answer.review_" + str(i) + "_judgment"]
			if (question_answer_worker.strip() == ""):
				workers_incomplete_hits.append(worker_id)
				has_incomplete_answers = 1
				break

			question_answer_gold_standard = row["Answer.{review_" + str(i) + "_student_model}"]
			question_text = row["Input.review_" + str(i) + "_text"]
			
			if question_answer_gold_standard == "Real":
				count_real += 1
				overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["text"] = question_text
				if question_answer_worker == "real":
					tp += 1

					student_real["correct"] += 1
					overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["correct"] += 1
					if (overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["correct"] == 5):
						overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["mistaken"] = 0
				elif question_answer_worker == "fake":
					fn += 1
					student_real["mistaken"] += 1
					overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["mistaken"] += 1
					if (overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
						overall_annotations_per_student_REAL[str(current_hit) + "_" + str(i)]["correct"] = 0
			else:
				## all negative gold standard answers
				count_fake += 1
				overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["text"] = question_text
				if question_answer_worker == "real":
					fp += 1
					student_fake["mistaken"] += 1
					overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["mistaken"] += 1
					if (overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
						overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["correct"] = 0

				elif question_answer_worker == "fake":
					student_fake["correct"] += 1
					tn += 1
					overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["correct"] += 1
					if (overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["correct"] == 5):
						overall_annotations_per_student_FAKE[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				# mistaken means annotated as real, while correct means annotated as fake
				if (question_answer_gold_standard == "WordRNN10"):
					count_WordRNN10 += 1
					overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_WordRNN10["mistaken"] += 1
						overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_WordRNN10["correct"] += 1
						overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_WordRNN10[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "WordRNN07"):
					count_WordRNN07 += 1
					overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_WordRNN07["mistaken"] += 1
						overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_WordRNN07["correct"] += 1
						overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_WordRNN07[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "WordRNN05"):
					count_WordRNN05 += 1
					overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_WordRNN05["mistaken"] += 1
						overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_WordRNN05["correct"] += 1
						overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_WordRNN05[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "GoogleLM"):
					count_GoogleLM += 1
					overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_GoogleLM["mistaken"] += 1
						overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_GoogleLM["correct"] += 1
						overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_GoogleLM[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "AttentionAC"):
					count_AttentionAC += 1
					overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_AttentionAC["mistaken"] += 1
						overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_AttentionAC["correct"] += 1
						overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_AttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "NoAttentionAC"):
					count_NoAttentionAC += 1
					overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_NoAttentionAC["mistaken"] += 1
						overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_NoAttentionAC["correct"] += 1
						overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_NoAttentionAC[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "SkipConnectionsAC"): 
					count_SkipConnections += 1
					overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_SkipConnections["mistaken"] += 1
						overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_SkipConnections["correct"] += 1
						overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_SkipConnections[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "MLESeqGAN"):
					count_MLESeqGAN += 1
					overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_MLESeqGAN["mistaken"] += 1
						overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_MLESeqGAN["correct"] += 1
						overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_MLESeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "SS"):
					count_SS += 1
					overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_SS["mistaken"] += 1
						overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_SS["correct"] += 1
						overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_SS[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "SeqGAN"):
					count_SeqGAN += 1
					overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_SeqGAN["mistaken"] += 1
						overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_SeqGAN["correct"] += 1
						overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_SeqGAN[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "RankGAN"):
					count_RankGAN += 1
					overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_RankGAN["mistaken"] += 1
						overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_RankGAN["correct"] += 1
						overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_RankGAN[str(current_hit) + "_" + str(i)]["mistaken"] = 0

				elif (question_answer_gold_standard == "LeakGAN"):
					count_LeakGAN += 1
					overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["text"] = question_text
					if question_answer_worker == "real":
						student_LeakGAN["mistaken"] += 1
						overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["mistaken"] += 1
						if (overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["mistaken"] == 5):
							overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["correct"] = 0
					elif question_answer_worker == "fake":
						student_LeakGAN["correct"] += 1
						overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["correct"] += 1
						if (overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["correct"] == 5):
							overall_annotations_per_student_LeakGAN[str(current_hit) + "_" + str(i)]["mistaken"] = 0

		
		if has_incomplete_answers == 0:
			# determine worker answers
			current_answers = dict()
			current_answers["tp"] = tp
			current_answers["fn"] = fn
			current_answers["tn"] = tn
			current_answers["fp"] = fp
			worker_answers[worker_id] = current_answers

			tp_all.append(tp)
			fn_all.append(fn)
			tn_all.append(tn)
			fp_all.append(fp)
			#print 50 * "*"
			#print "tp", tp, "fn", fn, "tn", tn, "fp", fp
			if (tp + fn + tn + fp != 20):
				print "!!!!!!!!!!!! ERROR"
			
		#print "tp_all", tp_all, "fn_all", fn_all, "tn_all", tn_all, "fp_all", fp_all
	"""
	print "student_WordRNN10", student_WordRNN10
	print count_WordRNN10_all
	print count_WordRNN10
	
	
	print "student_WordRNN07", student_WordRNN07
	print count_WordRNN07_all
	print count_WordRNN07

	
	print "student_WordRNN05", student_WordRNN05
	print count_WordRNN05_all
	print count_WordRNN05

	print "student_GoogleLM", student_GoogleLM
	print count_GoogleLM_all
	print count_GoogleLM

	
	print "WORKER ANSWERS", worker_answers
	

	## Make sure we have 180 distinct HITS and that for every HIT we have 5 annotations ##
	print "HIT DICTIONARY", hit_ids
	print "TOTAL HITS", len(hit_ids.keys())
	"""

	## Determine how many workers answered more than just 1 HIT
	## Determine if all workers are unique ##
	print "TOTAL WORKERS", len(worker_ids)
	print "DISTINCT WORKERS", len(set(worker_ids))
	## find duplicate workers
	duplicate_worker_ids = [(item, count) for item, count in collections.Counter(worker_ids).items() if count > 1]
	print "DUPLICATE WORKERS", duplicate_worker_ids
	print "LEN(DUPLICATE WORKERS)", len(duplicate_worker_ids)
	print "LEN(SET(DUPLICATE WORKERS))", len(set(duplicate_worker_ids))

	#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_duplicate.pickle", "wb") as handle:
	#	pickle.dump(set(duplicate_worker_ids), handle, protocol=pickle.HIGHEST_PROTOCOL)

	## Determine how many workers who participated in the pilot study have also participated in the real study
	print "TOTAL WORKERS PILOT STUDY", len(worker_ids_pilot_study)
	common_workers = set.intersection(worker_ids_pilot_study, set(worker_ids))
	print "COMMON WORKERS", common_workers
	print "LEN(COMMON WORKERS)", len(common_workers)

	## Determine how many workers failed to answer the gotcha question correctly
	print "BANNED WORKERS FAILED GOTCHA", set(banned_workers_failed_gotcha)
	print "LEN(BANNED WORKERS FAILED GOTCHA)", len(banned_workers_failed_gotcha)
	print "LEN(SET(BANNED WORKERS FAILED GOTCHA))", len(set(banned_workers_failed_gotcha))

	#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_failed_gotcha.pickle", "wb") as handle:
	#	pickle.dump(set(banned_workers_failed_gotcha), handle, protocol=pickle.HIGHEST_PROTOCOL)

	## Determine how many workers have incomplete hits
	print "WORKERS INCOMPLETE HITS", set(workers_incomplete_hits)
	print "LEN(WORKERS INCOMPLETE HITS)", len(workers_incomplete_hits)
	print "LEN(SET(WORKERS INCOMPLETE HITS))", len(set(workers_incomplete_hits))


	#with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/workers_incomplete_hits.pickle", "wb") as handle:
	#	pickle.dump(set(workers_incomplete_hits), handle, protocol=pickle.HIGHEST_PROTOCOL)

	## Make sure we have 180 distinct HITS and that for every HIT we have 5 annotations ##
	#print "HIT DICTIONARY", hit_ids
	print "TOTAL HITS", len(hit_ids.keys())
	not_completed_hits = {k: v for k, v in hit_ids.iteritems() if v < 5}
	print "NOT COMPLETED HITS", not_completed_hits
	print "LEN(NOT COMPLETED HITS)", len(not_completed_hits)


	## Determine the amount of time a worker spent annotating reviews ##
	print "MINUTES SPENT ANNOTATING", worker_time
	minutes_per_worker = list(worker_time.values())
	plot_annotation_time(minutes_per_worker)
	print "MIN", min(minutes_per_worker)
	print "MAX", max(minutes_per_worker)
	print "AVG", sum(minutes_per_worker)/len(minutes_per_worker)
	
	#tp_all = tp_all[:-5]
	#fn_all = fn_all[:-5]
	#tn_all = tn_all[:-5]
	#fp_all = fp_all[:-5]

	print "len(tp_all)", len(tp_all)
	print "len(fn_all)", len(fn_all)
	print "len(tn_all)", len(tn_all)
	print "len(fp_all)", len(fp_all)
	
	print tp_all[0], tp_all[1], tp_all[2]
	print fn_all[0], fn_all[1], fn_all[2]
	print tn_all[0], tn_all[1], tn_all[2]
	print fp_all[0], fp_all[1], fp_all[2]
	
	print "tp_all:", sum(tp_all)/len(tp_all), str((sum(tp_all)/len(tp_all))/10 * 100) + " %"
	print "fn_all:", sum(fn_all)/len(fn_all), str((sum(fn_all)/len(fn_all))/10 * 100) + " %"
	print "tn_all:", sum(tn_all)/len(tn_all), str((sum(tn_all)/len(tn_all))/10 * 100) + " %"
	print "fp_all:", sum(fp_all)/len(fp_all), str((sum(fp_all)/len(fp_all))/10 * 100) + " %"
	print "Human Accuracy all examples:", str(sum(tp_all + tn_all) / sum(tp_all + fn_all + tn_all + fp_all) * 100) + " %"
	

	print "student_WordRNN10", student_WordRNN10
	print "count_WordRNN10", count_WordRNN10
	print "Accuracy WordRNN10", student_WordRNN10["correct"] / count_WordRNN10

	print "student_WordRNN07", student_WordRNN07
	print "count_WordRNN07", count_WordRNN07
	print "Accuracy WordRNN07", student_WordRNN07["correct"] / count_WordRNN07

	print "student_WordRNN05", student_WordRNN05
	print "count_WordRNN05", count_WordRNN05
	print "Accuracy WordRNN05", student_WordRNN05["correct"] / count_WordRNN05

	print "student_SS", student_SS
	print "count_SS", count_SS
	print "Accuracy SS", student_SS["correct"] / count_SS

	print "student_GoogleLM", student_GoogleLM
	print "count_GoogleLM", count_GoogleLM
	print "Accuracy GoogleLM", student_GoogleLM["correct"] / count_GoogleLM

	print "student_AttentionAC", student_AttentionAC
	print "count_AttentionAC", count_AttentionAC
	print "Accuracy AttentionAC", student_AttentionAC["correct"] / count_AttentionAC

	print "student_NoAttentionAC", student_NoAttentionAC
	print "count_NoAttentionAC", count_NoAttentionAC
	print "Accuracy NoAttentionAC", student_NoAttentionAC["correct"] / count_NoAttentionAC

	print "student_SkipConnections", student_SkipConnections
	print "count_SkipConnections", count_SkipConnections
	print "Accuracy SkipConnections", student_SkipConnections["correct"] / count_SkipConnections

	print "student_MLESeqGAN", student_MLESeqGAN
	print "count_MLESeqGAN", count_MLESeqGAN
	print "Accuracy MLESeqGAN", student_MLESeqGAN["correct"] / count_MLESeqGAN

	print "student_SeqGAN", student_SeqGAN
	print "count_SeqGAN", count_SeqGAN
	print "Accuracy SeqGAN", student_SeqGAN["correct"] / count_SeqGAN

	print "student_RankGAN", student_RankGAN
	print "count_RankGAN", count_RankGAN
	print "Accuracy RankGAN", student_RankGAN["correct"] / count_RankGAN

	print "student_LeakGAN", student_LeakGAN
	print "count_LeakGAN", count_LeakGAN
	print "Accuracy LeakGAN", student_LeakGAN["correct"] / count_LeakGAN

	print "student_real", student_real
	print "count_real", count_real
	print "Accuracy real", student_real["correct"] / count_real

	print "student_fake", student_fake
	print "count_fake", count_fake
	print "Accuracy fake", student_fake["correct"] / count_fake


	#### STATISTICS PER EXAMPLE INCLUDING ALL 5 WORKERS: each dictionary is of the form "HitID_QuestionID: {"correct": 3, "mistaken": 2}" ####
	print "WordRNN10 overall_annotations_per_student", len(overall_annotations_per_student_WordRNN10)
	print "WordRNN07 overall_annotations_per_student", len(overall_annotations_per_student_WordRNN07)
	print "WordRNN05 overall_annotations_per_student", len(overall_annotations_per_student_WordRNN05)
	print "GoogleLM overall_annotations_per_student", len(overall_annotations_per_student_GoogleLM)
	print "AttentionAC overall_annotations_per_student", len(overall_annotations_per_student_AttentionAC)
	print "No AttentionAC overall_annotations_per_student", len(overall_annotations_per_student_NoAttentionAC)
	print "Skip Connections overall_annotations_per_student", len(overall_annotations_per_student_SkipConnections)
	print "MLESeqGAN overall_annotations_per_student", len(overall_annotations_per_student_MLESeqGAN)
	print "SS overall_annotations_per_student", len(overall_annotations_per_student_SS)
	print "SeqGAN overall_annotations_per_student", len(overall_annotations_per_student_SeqGAN)
	print "RankGAN overall_annotations_per_student", len(overall_annotations_per_student_RankGAN)
	print "LeakGAN overall_annotations_per_student", len(overall_annotations_per_student_LeakGAN)
	print "REAL overall_annotations_per_student", len(overall_annotations_per_student_REAL)
	print "FAKE overall_annotations_per_student", len(overall_annotations_per_student_FAKE)

	## COUNT HOW MANY SUM UP TO 5
	total_complete_items_wordrnn10 = 0
	for hitID_questionID in overall_annotations_per_student_WordRNN10:
		if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"] == 5:
			total_complete_items_wordrnn10 += 1
	print "TOTAL ITEMS WORDRNN10 5 ANNOTATORS:", total_complete_items_wordrnn10
	print "overall_annotations_per_student_WordRNN10", overall_annotations_per_student_WordRNN10
	print "student_WordRNN10", student_WordRNN10

	total_complete_items_wordrnn07 = 0
	for hitID_questionID in overall_annotations_per_student_WordRNN07:
		if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"] == 5:
			total_complete_items_wordrnn07 += 1
	print "TOTAL ITEMS WORDRNN07 5 ANNOTATORS:", total_complete_items_wordrnn07

	total_complete_items_wordrnn05 = 0
	for hitID_questionID in overall_annotations_per_student_WordRNN05:
		if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"] == 5:
			total_complete_items_wordrnn05 += 1
	print "TOTAL ITEMS WORDRNN05 5 ANNOTATORS:", total_complete_items_wordrnn05

	total_complete_items_googlelm = 0
	for hitID_questionID in overall_annotations_per_student_GoogleLM:
		if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] + overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"] == 5:
			total_complete_items_googlelm += 1
	print "TOTAL ITEMS GOOGLELM 5 ANNOTATORS:", total_complete_items_googlelm

	total_complete_items_attentionac = 0
	for hitID_questionID in overall_annotations_per_student_AttentionAC:
		if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"] == 5:
			total_complete_items_attentionac += 1
	print "TOTAL ITEMS ATTENTION AC 5 ANNOTATORS:", total_complete_items_attentionac

	total_complete_items_noattentionac = 0
	for hitID_questionID in overall_annotations_per_student_NoAttentionAC:
		if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"] == 5:
			total_complete_items_noattentionac += 1
	print "TOTAL ITEMS NO ATTENTION AC 5 ANNOTATORS:", total_complete_items_noattentionac

	total_complete_items_skipconnections = 0
	for hitID_questionID in overall_annotations_per_student_SkipConnections:
		if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] + overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"] == 5:
			total_complete_items_skipconnections += 1
	print "TOTAL ITEMS SKIP CONNECTIONS 5 ANNOTATORS:", total_complete_items_skipconnections

	total_complete_items_mleseqgan = 0
	for hitID_questionID in overall_annotations_per_student_MLESeqGAN:
		if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"] == 5:
			total_complete_items_mleseqgan += 1
	print "TOTAL ITEMS MLE SEQGAN 5 ANNOTATORS:", total_complete_items_mleseqgan

	total_complete_items_ss = 0
	for hitID_questionID in overall_annotations_per_student_SS:
		if overall_annotations_per_student_SS[hitID_questionID]["correct"] + overall_annotations_per_student_SS[hitID_questionID]["mistaken"] == 5:
			total_complete_items_ss += 1
	print "TOTAL ITEMS SS 5 ANNOTATORS:", total_complete_items_ss

	total_complete_items_seqgan = 0
	for hitID_questionID in overall_annotations_per_student_SeqGAN:
		if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"] == 5:
			total_complete_items_seqgan += 1
	print "TOTAL ITEMS SeqGAN 5 ANNOTATORS:", total_complete_items_seqgan

	total_complete_items_rankgan = 0
	for hitID_questionID in overall_annotations_per_student_RankGAN:
		if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] + overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"] == 5:
			total_complete_items_rankgan += 1
	print "TOTAL ITEMS RANKGAN 5 ANNOTATORS:", total_complete_items_rankgan

	total_complete_items_leakgan = 0
	for hitID_questionID in overall_annotations_per_student_LeakGAN:
		if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] + overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"] == 5:
			total_complete_items_leakgan += 1
	print "TOTAL ITEMS LEAKGAN 5 ANNOTATORS:", total_complete_items_leakgan

	total_complete_items_real = 0
	for hitID_questionID in overall_annotations_per_student_REAL:
		if overall_annotations_per_student_REAL[hitID_questionID]["correct"] + overall_annotations_per_student_REAL[hitID_questionID]["mistaken"] == 5:
			total_complete_items_real += 1
	print "TOTAL ITEMS REAL 5 ANNOTATORS:", total_complete_items_real

	total_complete_items_fake = 0
	for hitID_questionID in overall_annotations_per_student_FAKE:
		if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] + overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"] == 5:
			total_complete_items_fake += 1
	print "TOTAL ITEMS REAL 5 ANNOTATORS:", total_complete_items_fake


	#### COMPUTE INTER_ANNOTATOR AGREEMENT WHEN EACH EXAMPLE IS ANNOTATED BY 5 WORKERS ####
	
	fleis_kappa_matrix_REAL = []
	for hitID_questionID in overall_annotations_per_student_REAL:
		if overall_annotations_per_student_REAL[hitID_questionID]["correct"] + overall_annotations_per_student_REAL[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_REAL.append([overall_annotations_per_student_REAL[hitID_questionID]["mistaken"], overall_annotations_per_student_REAL[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix REAL", len(fleis_kappa_matrix_REAL)

	fleis_kappa_matrix_FAKE = []
	for hitID_questionID in overall_annotations_per_student_FAKE:
		if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] + overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_FAKE.append([overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"], overall_annotations_per_student_FAKE[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix FAKE", len(fleis_kappa_matrix_FAKE)

	fleis_kappa_matrix_ALL = fleis_kappa_matrix_FAKE + fleis_kappa_matrix_REAL
	print "Fleis Kappa matrix ALL", len(fleis_kappa_matrix_ALL)

	fleis_kappa_matrix_WordRNN10 = []
	for hitID_questionID in overall_annotations_per_student_WordRNN10:
		if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_WordRNN10.append([overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"], overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix WordRNN10", len(fleis_kappa_matrix_WordRNN10)

	fleis_kappa_matrix_WordRNN07 = []
	for hitID_questionID in overall_annotations_per_student_WordRNN07:
		if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_WordRNN07.append([overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"], overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix WordRNN07", len(fleis_kappa_matrix_WordRNN07)

	fleis_kappa_matrix_WordRNN05 = []
	for hitID_questionID in overall_annotations_per_student_WordRNN05:
		if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_WordRNN05.append([overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"], overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix WordRNN05", len(fleis_kappa_matrix_WordRNN05)

	fleis_kappa_matrix_GoogleLM = []
	for hitID_questionID in overall_annotations_per_student_GoogleLM:
		if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] + overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_GoogleLM.append([overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"], overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix GoogleLM", len(fleis_kappa_matrix_GoogleLM)

	fleis_kappa_matrix_AttentionAC = []
	for hitID_questionID in overall_annotations_per_student_AttentionAC:
		if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_AttentionAC.append([overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"], overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix AttentionAC", len(fleis_kappa_matrix_AttentionAC)

	fleis_kappa_matrix_NoAttentionAC = []
	for hitID_questionID in overall_annotations_per_student_NoAttentionAC:
		if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_NoAttentionAC.append([overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"], overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix NoAttentionAC", len(fleis_kappa_matrix_NoAttentionAC)	

	fleis_kappa_matrix_SkipConnections = []
	for hitID_questionID in overall_annotations_per_student_SkipConnections:
		if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] + overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_SkipConnections.append([overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"], overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix SkipConnections", len(fleis_kappa_matrix_SkipConnections)

	fleis_kappa_matrix_MLESeqGAN = []
	for hitID_questionID in overall_annotations_per_student_MLESeqGAN:
		if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_MLESeqGAN.append([overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"], overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix MLESeqgan", len(fleis_kappa_matrix_MLESeqGAN)

	fleis_kappa_matrix_SS = []
	for hitID_questionID in overall_annotations_per_student_SS:
		if overall_annotations_per_student_SS[hitID_questionID]["correct"] + overall_annotations_per_student_SS[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_SS.append([overall_annotations_per_student_SS[hitID_questionID]["mistaken"], overall_annotations_per_student_SS[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix SS", len(fleis_kappa_matrix_SS)

	fleis_kappa_matrix_SeqGAN = []
	for hitID_questionID in overall_annotations_per_student_SeqGAN:
		if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_SeqGAN.append([overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"], overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix SeqGAN", len(fleis_kappa_matrix_SeqGAN)

	fleis_kappa_matrix_RankGAN = []
	for hitID_questionID in overall_annotations_per_student_RankGAN:
		if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] + overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_RankGAN.append([overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"], overall_annotations_per_student_RankGAN[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix RankGAN", len(fleis_kappa_matrix_RankGAN)

	fleis_kappa_matrix_LeakGAN = []
	for hitID_questionID in overall_annotations_per_student_LeakGAN:
		if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] + overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"] == 5:
			fleis_kappa_matrix_LeakGAN.append([overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"], overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"]])
	print "Fleis Kappa matrix LeakGAN", len(fleis_kappa_matrix_LeakGAN)

	
	# save pilot study worker ids to disk
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_ALL.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_ALL, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_REAL.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_REAL, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_FAKE.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_FAKE, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_WordRNN10.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_WordRNN10, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_WordRNN07.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_WordRNN07, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_WordRNN05.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_WordRNN05, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_GoogleLM.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_GoogleLM, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_AttentionAC.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_AttentionAC, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_NoAttentionAC.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_NoAttentionAC, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_SkipConnections.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_SkipConnections, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_MLESeqGAN.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_MLESeqGAN, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_SS.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_SS, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_SeqGAN.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_SeqGAN, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_RankGAN.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_RankGAN, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/fleis_kappa_matrix_LeakGAN.pickle", "wb") as handle:
		pickle.dump(fleis_kappa_matrix_LeakGAN, handle, protocol=pickle.HIGHEST_PROTOCOL)
	
	

	###### Consider the majority vote #######
	majority_vote_WordRNN10 = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_WordRNN10:
		if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] > overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"]:
				majority_vote_WordRNN10["fake"] += 1
			else:
				majority_vote_WordRNN10["true"] += 1
	print "MAJORITY VOTE WORDRNN10", majority_vote_WordRNN10
	print "MAJORITY VOTE WORDRNN10 ACCURACY:", majority_vote_WordRNN10["fake"] / (majority_vote_WordRNN10["fake"] + majority_vote_WordRNN10["true"])

	majority_vote_WordRNN07 = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_WordRNN07:
		if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] > overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"]:
				majority_vote_WordRNN07["fake"] += 1
			else:
				majority_vote_WordRNN07["true"] += 1
	print "MAJORITY VOTE WORDRNN07", majority_vote_WordRNN07
	print "MAJORITY VOTE WORDRNN07 ACCURACY:", majority_vote_WordRNN07["fake"] / (majority_vote_WordRNN07["fake"] + majority_vote_WordRNN07["true"])

	majority_vote_WordRNN05 = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_WordRNN05:
		if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] > overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"]:
				majority_vote_WordRNN05["fake"] += 1
			else:
				majority_vote_WordRNN05["true"] += 1
	print "MAJORITY VOTE WORDRNN05", majority_vote_WordRNN05
	print "MAJORITY VOTE WORDRNN05 ACCURACY:", majority_vote_WordRNN05["fake"] / (majority_vote_WordRNN05["fake"] + majority_vote_WordRNN05["true"])

	majority_vote_GoogleLM = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_GoogleLM:
		if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] + overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] > overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"]:
				majority_vote_GoogleLM["fake"] += 1
			else:
				majority_vote_GoogleLM["true"] += 1
	print "MAJORITY VOTE GoogleLM", majority_vote_GoogleLM
	print "MAJORITY VOTE GoogleLM ACCURACY:", majority_vote_GoogleLM["fake"] / (majority_vote_GoogleLM["fake"] + majority_vote_GoogleLM["true"])

	majority_vote_AttentionAC = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_AttentionAC:
		if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] > overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"]:
				majority_vote_AttentionAC["fake"] += 1
			else:
				majority_vote_AttentionAC["true"] += 1
	print "MAJORITY VOTE AttentionAC", majority_vote_AttentionAC
	print "MAJORITY VOTE AttentionAC ACCURACY:", majority_vote_AttentionAC["fake"] / (majority_vote_AttentionAC["fake"] + majority_vote_AttentionAC["true"])

	majority_vote_NoAttentionAC = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_NoAttentionAC:
		if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] > overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"]:
				majority_vote_NoAttentionAC["fake"] += 1
			else:
				majority_vote_NoAttentionAC["true"] += 1
	print "MAJORITY VOTE NoAttentionAC", majority_vote_NoAttentionAC
	print "MAJORITY VOTE NoAttentionAC ACCURACY:", majority_vote_NoAttentionAC["fake"] / (majority_vote_NoAttentionAC["fake"] + majority_vote_NoAttentionAC["true"])

	majority_vote_SkipConnections = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_SkipConnections:
		if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] + overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] > overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"]:
				majority_vote_SkipConnections["fake"] += 1
			else:
				majority_vote_SkipConnections["true"] += 1
	print "MAJORITY VOTE SkipConnections", majority_vote_SkipConnections
	print "MAJORITY VOTE SkipConnections ACCURACY:", majority_vote_SkipConnections["fake"] / (majority_vote_SkipConnections["fake"] + majority_vote_SkipConnections["true"])

	majority_vote_MLESeqGAN = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_MLESeqGAN:
		if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] > overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"]:
				majority_vote_MLESeqGAN["fake"] += 1
			else:
				majority_vote_MLESeqGAN["true"] += 1
	print "MAJORITY VOTE MLE Seqgan", majority_vote_MLESeqGAN
	print "MAJORITY VOTE MLE Seqgan ACCURACY:", majority_vote_MLESeqGAN["fake"] / (majority_vote_MLESeqGAN["fake"] + majority_vote_MLESeqGAN["true"])

	majority_vote_SS = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_SS:
		if overall_annotations_per_student_SS[hitID_questionID]["correct"] + overall_annotations_per_student_SS[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_SS[hitID_questionID]["correct"] > overall_annotations_per_student_SS[hitID_questionID]["mistaken"]:
				majority_vote_SS["fake"] += 1
			else:
				majority_vote_SS["true"] += 1
	print "MAJORITY VOTE SS", majority_vote_SS
	print "MAJORITY VOTE SS ACCURACY:", majority_vote_SS["fake"] / (majority_vote_SS["fake"] + majority_vote_SS["true"])

	majority_vote_SeqGAN = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_SeqGAN:
		if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] > overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"]:
				majority_vote_SeqGAN["fake"] += 1
			else:
				majority_vote_SeqGAN["true"] += 1
	print "MAJORITY VOTE SeqGAN", majority_vote_SeqGAN
	print "MAJORITY VOTE SeqGAN ACCURACY:", majority_vote_SeqGAN["fake"] / (majority_vote_SeqGAN["fake"] + majority_vote_SeqGAN["true"])

	majority_vote_RankGAN = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_RankGAN:
		if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] + overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] > overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"]:
				majority_vote_RankGAN["fake"] += 1
			else:
				majority_vote_RankGAN["true"] += 1
	print "MAJORITY VOTE RankGAN", majority_vote_RankGAN
	print "MAJORITY VOTE RankGAN ACCURACY:", majority_vote_RankGAN["fake"] / (majority_vote_RankGAN["fake"] + majority_vote_RankGAN["true"])

	majority_vote_LeakGAN = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_LeakGAN:
		if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] + overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] > overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"]:
				majority_vote_LeakGAN["fake"] += 1
			else:
				majority_vote_LeakGAN["true"] += 1
	print "MAJORITY VOTE LeakGAN", majority_vote_LeakGAN
	print "MAJORITY VOTE LeakGAN ACCURACY:", majority_vote_LeakGAN["fake"] / (majority_vote_LeakGAN["fake"] + majority_vote_LeakGAN["true"])

	majority_vote_REAL = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_REAL:
		if overall_annotations_per_student_REAL[hitID_questionID]["correct"] + overall_annotations_per_student_REAL[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_REAL[hitID_questionID]["correct"] > overall_annotations_per_student_REAL[hitID_questionID]["mistaken"]:
				majority_vote_REAL["true"] += 1
			else:
				majority_vote_REAL["fake"] += 1
	print "MAJORITY VOTE REAL", majority_vote_REAL

	print "tp majority vote", majority_vote_REAL["true"] / (majority_vote_REAL["true"] + majority_vote_REAL["fake"]) 
	print "fn majority vote", majority_vote_REAL["fake"] / (majority_vote_REAL["true"] + majority_vote_REAL["fake"])

	majority_vote_FAKE = defaultdict(int)
	for hitID_questionID in overall_annotations_per_student_FAKE:
		if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] + overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] > overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"]:
				majority_vote_FAKE["fake"] += 1
			else:
				majority_vote_FAKE["true"] += 1
	print "MAJORITY VOTE FAKE", majority_vote_FAKE

	print "tn majority vote", majority_vote_FAKE["fake"] / (majority_vote_FAKE["fake"] + majority_vote_FAKE["true"])
	print "fn majority vote", majority_vote_FAKE["true"] / (majority_vote_FAKE["fake"] + majority_vote_FAKE["true"])

	print "Accuracy majority vote", (majority_vote_REAL["true"] + majority_vote_FAKE["fake"])/(majority_vote_REAL["true"] + majority_vote_REAL["fake"] + majority_vote_FAKE["fake"] +  majority_vote_FAKE["true"])


	###########  ASK DISCRIMINATORS TO PREDICT ON THE ANNOTATED EXAMPLES USING THE HUMAN JUDGEMENTS AS GROUND TRUTH  ###############
	
	student_WordRNN10_annotated_examples = []
	student_WordRNN10_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_WordRNN10:
		question_text = overall_annotations_per_student_WordRNN10[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"] > overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_WordRNN10_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_WordRNN10[hitID_questionID]["correct"]):
			human_label = 0
			student_WordRNN10_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_WordRNN10[hitID_questionID]["mistaken"]):
			human_label = 1
			student_WordRNN10_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_WordRNN10_annotated_examples)", len(student_WordRNN10_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_WordRNN10_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_WordRNN10_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_WordRNN10_annotated_examples_all_annotations)", len(student_WordRNN10_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_WordRNN10_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_WordRNN10_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_WordRNN07_annotated_examples = []
	student_WordRNN07_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_WordRNN07:
		question_text = overall_annotations_per_student_WordRNN07[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"] > overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_WordRNN07_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_WordRNN07[hitID_questionID]["correct"]):
			human_label = 0
			student_WordRNN07_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_WordRNN07[hitID_questionID]["mistaken"]):
			human_label = 1
			student_WordRNN07_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_WordRNN07_annotated_examples)", len(student_WordRNN07_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_WordRNN07_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_WordRNN07_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_WordRNN07_annotated_examples_all_annotations)", len(student_WordRNN07_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_WordRNN07_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_WordRNN07_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_WordRNN05_annotated_examples = []
	student_WordRNN05_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_WordRNN05:
		question_text = overall_annotations_per_student_WordRNN05[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] + overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"] > overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_WordRNN05_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_WordRNN05[hitID_questionID]["correct"]):
			human_label = 0
			student_WordRNN05_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_WordRNN05[hitID_questionID]["mistaken"]):
			human_label = 1
			student_WordRNN05_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_WordRNN05_annotated_examples)", len(student_WordRNN05_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_WordRNN05_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_WordRNN05_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_WordRNN05_annotated_examples_all_annotations)", len(student_WordRNN05_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_WordRNN05_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_WordRNN05_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_GoogleLM_annotated_examples = []
	student_GoogleLM_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_GoogleLM:
		question_text = overall_annotations_per_student_GoogleLM[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] + overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"] > overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_GoogleLM_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_GoogleLM[hitID_questionID]["correct"]):
			human_label = 0
			student_GoogleLM_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_GoogleLM[hitID_questionID]["mistaken"]):
			human_label = 1
			student_GoogleLM_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_GoogleLM_annotated_examples)", len(student_GoogleLM_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_GoogleLM_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_GoogleLM_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_GoogleLM_annotated_examples_all_annotations)", len(student_GoogleLM_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_GoogleLM_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_GoogleLM_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_AttentionAC_annotated_examples = []
	student_AttentionAC_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_AttentionAC:
		question_text = overall_annotations_per_student_AttentionAC[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"] > overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_AttentionAC_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_AttentionAC[hitID_questionID]["correct"]):
			human_label = 0
			student_AttentionAC_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_AttentionAC[hitID_questionID]["mistaken"]):
			human_label = 1
			student_AttentionAC_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_AttentionAC_annotated_examples)", len(student_AttentionAC_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_AttentionAC_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_AttentionAC_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_AttentionAC_annotated_examples_all_annotations)", len(student_AttentionAC_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_AttentionAC_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_AttentionAC_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_NoAttentionAC_annotated_examples = []
	student_NoAttentionAC_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_NoAttentionAC:
		question_text = overall_annotations_per_student_NoAttentionAC[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] + overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"] > overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_NoAttentionAC_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_NoAttentionAC[hitID_questionID]["correct"]):
			human_label = 0
			student_NoAttentionAC_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_NoAttentionAC[hitID_questionID]["mistaken"]):
			human_label = 1
			student_NoAttentionAC_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_NoAttentionAC_annotated_examples)", len(student_NoAttentionAC_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_NoAttentionAC_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_NoAttentionAC_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_NoAttentionAC_annotated_examples_all_annotations)", len(student_NoAttentionAC_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_NoAttentionAC_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_NoAttentionAC_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_SkipConnections_annotated_examples = []
	student_SkipConnections_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_SkipConnections:
		question_text = overall_annotations_per_student_SkipConnections[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] + overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"] > overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_SkipConnections_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_SkipConnections[hitID_questionID]["correct"]):
			human_label = 0
			student_SkipConnections_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_SkipConnections[hitID_questionID]["mistaken"]):
			human_label = 1
			student_SkipConnections_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_SkipConnections_annotated_examples)", len(student_SkipConnections_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_SkipConnections_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_SkipConnections_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_NoAttentionAC_annotated_examples_all_annotations)", len(student_SkipConnections_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_SkipConnections_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_SkipConnections_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)


	student_MLESeqGAN_annotated_examples = []
	student_MLESeqGAN_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_MLESeqGAN:
		question_text = overall_annotations_per_student_MLESeqGAN[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"] > overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_MLESeqGAN_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_MLESeqGAN[hitID_questionID]["correct"]):
			human_label = 0
			student_MLESeqGAN_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_MLESeqGAN[hitID_questionID]["mistaken"]):
			human_label = 1
			student_MLESeqGAN_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_MLESeqGAN_annotated_examples)", len(student_MLESeqGAN_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_MLESeqGAN_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_MLESeqGAN_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_MLESeqGAN_annotated_examples_all_annotations)", len(student_MLESeqGAN_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_MLESeqGAN_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_MLESeqGAN_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_SS_annotated_examples = []
	student_SS_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_SS:
		question_text = overall_annotations_per_student_SS[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_SS[hitID_questionID]["correct"] + overall_annotations_per_student_SS[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_SS[hitID_questionID]["correct"] > overall_annotations_per_student_SS[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_SS_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_SS[hitID_questionID]["correct"]):
			human_label = 0
			student_SS_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_SS[hitID_questionID]["mistaken"]):
			human_label = 1
			student_SS_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_SS_annotated_examples)", len(student_SS_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_SS_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_SS_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_SS_annotated_examples_all_annotations)", len(student_SS_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_SS_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_SS_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_SeqGAN_annotated_examples = []
	student_SeqGAN_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_SeqGAN:
		question_text = overall_annotations_per_student_SeqGAN[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] + overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"] > overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_SeqGAN_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_SeqGAN[hitID_questionID]["correct"]):
			human_label = 0
			student_SeqGAN_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_SeqGAN[hitID_questionID]["mistaken"]):
			human_label = 1
			student_SeqGAN_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_SeqGAN_annotated_examples)", len(student_SeqGAN_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_SeqGAN_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_SeqGAN_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_SeqGAN_annotated_examples_all_annotations)", len(student_SeqGAN_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_SeqGAN_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_SeqGAN_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)


	student_RankGAN_annotated_examples = []
	student_RankGAN_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_RankGAN:
		question_text = overall_annotations_per_student_RankGAN[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] + overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_RankGAN[hitID_questionID]["correct"] > overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_RankGAN_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_RankGAN[hitID_questionID]["correct"]):
			human_label = 0
			student_RankGAN_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_RankGAN[hitID_questionID]["mistaken"]):
			human_label = 1
			student_RankGAN_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_RankGAN_annotated_examples)", len(student_RankGAN_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_RankGAN_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_RankGAN_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_RankGAN_annotated_examples_all_annotations)", len(student_RankGAN_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_RankGAN_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_RankGAN_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_LeakGAN_annotated_examples = []
	student_LeakGAN_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_LeakGAN:
		question_text = overall_annotations_per_student_LeakGAN[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] + overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"] > overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_LeakGAN_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_LeakGAN[hitID_questionID]["correct"]):
			human_label = 0
			student_LeakGAN_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_LeakGAN[hitID_questionID]["mistaken"]):
			human_label = 1
			student_LeakGAN_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_LeakGAN_annotated_examples)", len(student_LeakGAN_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_LeakGAN_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_LeakGAN_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_LeakGAN_annotated_examples_all_annotations)", len(student_LeakGAN_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_LeakGAN_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_LeakGAN_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_REAL_annotated_examples = []
	student_REAL_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_REAL:
		question_text = overall_annotations_per_student_REAL[hitID_questionID]["text"]
		true_label = 1
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_REAL[hitID_questionID]["correct"] + overall_annotations_per_student_REAL[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_REAL[hitID_questionID]["correct"] > overall_annotations_per_student_REAL[hitID_questionID]["mistaken"]:
				human_label = 1
			else:
				human_label = 0
		student_REAL_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_REAL[hitID_questionID]["correct"]):
			human_label = 0
			student_REAL_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_REAL[hitID_questionID]["mistaken"]):
			human_label = 1
			student_REAL_annotated_examples_all_annotations.append((question_text, true_label, human_label))

	print "LEN(student_REAL_annotated_examples)", len(student_REAL_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_REAL_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_REAL_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_REAL_annotated_examples_all_annotations)", len(student_REAL_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_REAL_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_REAL_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)

	student_FAKE_annotated_examples = []
	student_FAKE_annotated_examples_all_annotations = []
	for hitID_questionID in overall_annotations_per_student_FAKE:
		question_text = overall_annotations_per_student_FAKE[hitID_questionID]["text"]
		true_label = 0
		#print "QUESTION TEXT", question_text
		if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] + overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"] == 5:
			if overall_annotations_per_student_FAKE[hitID_questionID]["correct"] > overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"]:
				human_label = 0
			else:
				human_label = 1
		student_FAKE_annotated_examples.append((question_text, true_label, human_label))

		for k in range(overall_annotations_per_student_FAKE[hitID_questionID]["correct"]):
			human_label = 0
			student_FAKE_annotated_examples_all_annotations.append((question_text, true_label, human_label))
		for j in range(overall_annotations_per_student_FAKE[hitID_questionID]["mistaken"]):
			human_label = 1
			student_FAKE_annotated_examples_all_annotations.append((question_text, true_label, human_label))
	print "LEN(student_FAKE_annotated_examples)", len(student_FAKE_annotated_examples)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels/student_FAKE_annotated_examples.pickle", "wb") as handle:
		pickle.dump(student_FAKE_annotated_examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
	print "LEN(student_FAKE_annotated_examples_all_annotations)", len(student_FAKE_annotated_examples_all_annotations)
	with open("/srv/disk01/ggarbace/EvaluationProj/fake_text/saved_models_normalized/mturk_real_study/annotated_examples_human_labels_all_annotations/student_FAKE_annotated_examples_all_annotations.pickle", "wb") as handle:
		pickle.dump(student_FAKE_annotated_examples_all_annotations, handle, protocol=pickle.HIGHEST_PROTOCOL)




