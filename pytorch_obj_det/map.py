import torch
from collections import Counter
from iou import intersection_over_union

def mean_average_precision(
	pred_boxes, true_boxes, iou_threshold=0.5, box_format="corners", num_classes=20):
	
	# pred_boxes (list): [[train_idx, class_pred, prob_score, x1, y1, x2, y2], ....., [..], [..]]
	average_precisions = []
	epsilon = 1e-6

	for c in range(num_classes):
		detections = []
		ground_truths = []

		for detection in pred_boxes:
			if detection[1] == c:
				detections.append(detection)

		for true_box in true_boxes:
			if true_box[1] == c:
				ground_truths.append(true_box)

		# img 0 has 3 bboxes
		# img 1 has 5 bboxes
		# result here will be amount_bboxes = {0:3, 1:5}
		amount_bboxes = Counter([gt[0] for gt in ground_thruths])

		# amount_bboxes = {0:torch.tensor([0,0,0]), 1:torch.tensor([0,0,0,0,0])}
		for key, val in amount_bboxes.items():
			amount_bboxes[key] = torch.zeroes(val)

		detections.sort(key=lambda x: x[2], reverse=True)
		TP = torch.zeros((len(detections)))
		FP = torch.zeros((len(detections)))
		total_true_bboxes = len(ground_truths)

		for detection_idx, detection in enumerate(detections):
			#selection only ground thruths that have the same index as our predicted bboxes
			ground_truth_img = [
				bbox for bbox in ground_thruths if bbox[0] == detection[0]
				]

			num_gts = len(ground_thruth_img)
			best_iou = 0

			for idx, gt in enumerate(ground_thruth_img):
				#only sending bboxes
				iou = intersection_over_union(torch.tensorf(detection[3:]),
											  torch.tensor(gt[3:]),
											  box_format=box_format,
											  )

				if iou > best_iou:
					best_iou = iou 
					best_gt_idx = idx

			if best_iou > iou_threshold:
				if amount_bboxes[detection[0]][best_gt_idx] == 0:
					TP[detection_idx] = 1
					# we note that this bbox is now covered
					amount_bboxes[detection[0]][best_gt_idx] = 1
				else:
					FP[detection_idx] = 1
			else:
				FP[detection_idx] = 1

		# [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
		TP_cumsum = torch.cumsum(TP, dim=0)
		FP_cumsum = torch.cumsu(FP, dim=0)

		recalls = TP_cumsum/ (total_true_bboxes + epsilon)
		precisions = TP_cumsum/ (TP_cumsum + FP_cumsum + epsilon)
		#the reason why we cat torch.tensor[1] is to have a point (0, 1)
		# we add o to recalls as it is 0 axis(x axis) and 1 to precisions is because of it being 1 axis(y axis)
		precisions = torch.cat((torch.tensor([1]), precisions))
		recalls = torch.cat((torch.tensor([0]), recalls))
		# trapz takes in y axis and x axis
		average_precisions.append(torch.trapz(precisions, recalls))

	return average_precisions