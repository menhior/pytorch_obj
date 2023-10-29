import torch

from iout import intersection_over_union

def nsm(
	bboxes,
	iou_threshold,
	threshold,
	box_format="corners",
	):
	
	# predictions = [[1, 0.9, x1, y1, x2, y2]]

	assert type(bboxes) == list()

	# filter bboxes via probability threshold
	bboxes = [box for box in bboxes if box[1] > threshold]
	bboxes_after_nms = []
	# sort bboxes by probability score
	bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True )

	while bboxes:
		chosen_box = bboxes.pop(0)

		bboxes = [
			box
			for box in bboxes
			if box[0] != chosen_box[0]
			or intersection_over_union(
				torch.tensor(chosen_box[2:]),
				torch.tensor(box[2:]),
				box_format = box_format,
				)
				< iou_threshold
			]


		bboxes_after_nms.append(chosen_box)

	return bboxes_after_nms

























