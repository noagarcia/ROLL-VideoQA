import numpy as np


def boxes_intersect(bbox1_left, bbox1_top, bbox1_right, bbox1_bottom,
                    bbox2_left, bbox2_top, bbox2_right, bbox2_bottom):
    left = np.maximum(bbox1_left, bbox2_left)
    top = np.maximum(bbox1_top, bbox2_top)
    right = np.minimum(bbox1_right, bbox2_right)
    bottom = np.minimum(bbox1_bottom, bbox2_bottom)
    return (left, top, right, bottom)


def box_area(left, top, right, bottom):
    area = (right - left + 1) * (bottom - top + 1)
    return area


def box_centroid(left, top, right, bottom):
    x = left + ((right - left + 1) / 2)
    y = top + ((bottom - top + 1) / 2)
    return (x, y)


def connected_components(connGraph):

    result = []
    nodes = set(connGraph.keys())
    while nodes:
        n = nodes.pop()

        # Set to contain nodes in this connected group
        group = {n}

        # Find neighbours, add neighbours to queue and remove from global set.
        queue = [n]
        while queue:
            n = queue.pop(0)
            neighbors = set(connGraph[n])
            neighbors.difference_update(group)
            nodes.difference_update(neighbors)
            group.update(neighbors)
            queue.extend(neighbors)

        # Add the group to the list of groups.
        result.append(group)

    return result


def select_top_triplets(res, topk):

    # in oi_all_rel some images have no dets
    if res['prd_scores'] is None:
        det_boxes_s_top = np.zeros((0, 4), dtype=np.float32)
        det_boxes_o_top = np.zeros((0, 4), dtype=np.float32)
        det_labels_s_top = np.zeros(0, dtype=np.int32)
        det_labels_p_top = np.zeros(0, dtype=np.int32)
        det_labels_o_top = np.zeros(0, dtype=np.int32)
        det_scores_top = np.zeros(0, dtype=np.float32)
    else:
        det_boxes_sbj = res['sbj_boxes']  # (#num_rel, 4)
        det_boxes_obj = res['obj_boxes']  # (#num_rel, 4)
        det_labels_sbj = res['sbj_labels']  # (#num_rel,)
        det_labels_obj = res['obj_labels']  # (#num_rel,)
        det_scores_sbj = res['sbj_scores']  # (#num_rel,)
        det_scores_obj = res['obj_scores']  # (#num_rel,)
        det_scores_prd = res['prd_scores'][:, 1:]

        det_labels_prd = np.argsort(-det_scores_prd, axis=1)
        det_scores_prd = -np.sort(-det_scores_prd, axis=1)

        det_scores_so = det_scores_sbj * det_scores_obj
        det_scores_spo = det_scores_so[:, None] * det_scores_prd

        det_scores_inds = np.column_stack(np.unravel_index(np.argsort(-det_scores_spo.ravel()), det_scores_spo.shape))[:topk]
        det_scores_top = det_scores_spo[det_scores_inds[:, 0], det_scores_inds[:, 1]]
        det_boxes_so_top = np.hstack(
            (det_boxes_sbj[det_scores_inds[:, 0]], det_boxes_obj[det_scores_inds[:, 0]]))
        det_labels_p_top = det_labels_prd[det_scores_inds[:, 0], det_scores_inds[:, 1]]
        det_labels_spo_top = np.vstack(
            (det_labels_sbj[det_scores_inds[:, 0]], det_labels_p_top,
             det_labels_obj[det_scores_inds[:, 0]])).transpose()

        det_boxes_s_top = det_boxes_so_top[:, :4]
        det_boxes_o_top = det_boxes_so_top[:, 4:]
        det_labels_s_top = det_labels_spo_top[:, 0]
        det_labels_p_top = det_labels_spo_top[:, 1]
        det_labels_o_top = det_labels_spo_top[:, 2]

    topk_dets = dict(image=res['image'],
                          det_boxes_s_top=det_boxes_s_top,
                          det_boxes_o_top=det_boxes_o_top,
                          det_labels_s_top=det_labels_s_top,
                          det_labels_p_top=det_labels_p_top,
                          det_labels_o_top=det_labels_o_top,
                          det_scores_top=det_scores_top)

    return topk_dets