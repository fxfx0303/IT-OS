from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from utils import box_ops
from utils.misc import (NestedTensor, interpolate)

import utils.dist as dist

# from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
#                        accuracy, get_world_size, interpolate,
#                        is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
# from .segmentation import (Modelsegm, PostProcessSegm,
#                            dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class Model(nn.Module):
    def __init__(
            self,
            backbone,
            transformer,
            num_classes,
            num_frames,
            num_queries,
            aux_loss=False,
            contrastive_hdim=64,
            contrastive_loss=False,
            contrastive_align_loss=False,
            predict_final=False,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.hidden_dim = hidden_dim
        # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.class_embed_new = nn.Linear(hidden_dim, num_classes + 1)
        self.isfinal_embed = nn.Linear(hidden_dim, 1) if predict_final else None
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_new = nn.Embedding(num_queries, hidden_dim)
        self.num_frames = num_frames

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

###################
# contrastive loss
###################
        # self.contrastive_loss = contrastive_loss
        # if contrastive_loss:
        #     self.contrastive_projection_image = nn.Linear(hidden_dim, contrastive_hdim, bias=False)
        #     self.contrastive_projection_text = nn.Linear(
        #         self.transformer.text_encoder.config.hidden_size, contrastive_hdim, bias=False
        #     )
        # self.contrastive_align_loss = contrastive_align_loss
        # if contrastive_align_loss:
        #     self.contrastive_align_projection_image = nn.Linear(hidden_dim, contrastive_hdim)
        #     self.contrastive_align_projection_text = nn.Linear(hidden_dim, contrastive_hdim)

    def forward(self, samples: NestedTensor, captions, encode_and_save=True, memory_cache=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: image sequences, of shape [num_frames x 3 x H x W]
               - samples.mask: a binary mask of shape [num_frames x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        if not isinstance(samples, NestedTensor):
            samples = NestedTensor.from_tensor_list(samples)

        # if not isinstance(samples, NestedTensor):
        #     samples = nested_tensor_from_tensor_list(samples)

        # # if encode_and_save:
        # assert memory_cache is None
        # # moved the frame to batch dimension for computation efficiency
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        query_embed = self.query_embed_new.weight
        pos = pos[-1]
        src_proj = self.input_proj(src)
        n, c, h, w = src_proj.shape
        assert mask is not None
        src_proj = src_proj.reshape(n // self.num_frames, self.num_frames, c, h, w).permute(0, 2, 1, 3, 4).flatten(
            -2)
        mask = mask.reshape(n // self.num_frames, self.num_frames, h * w)
        pos = pos.permute(0, 2, 1, 3, 4).flatten(-2)

        hs = self.transformer(
            src_proj,
            mask,
            query_embed,
            pos,
            captions,
        )

        # src_proj = self.input_proj(src)
        # n, c, h, w = src_proj.shape
        # assert mask is not None
        # src_proj = src_proj.reshape(n // self.num_frames, self.num_frames, c, h, w).permute(0, 2, 1, 3, 4).flatten(-2)
        # mask = mask.reshape(n // self.num_frames, self.num_frames, h * w)
        # pos = pos.permute(0, 2, 1, 3, 4).flatten(-2)
        # hs = self.transformer(src_proj, mask, self.query_embed.weight, pos)[0]

        out = {}
        outputs_class = self.class_embed_new(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        out.update(
            {
                "pred_logits": outputs_class[-1],
                "pred_boxes": outputs_coord[-1],
            }
        )
        outputs_isfinal = None
        if self.isfinal_embed is not None:
            outputs_isfinal = self.isfinal_embed(hs)
            out["pred_isfinal"] = outputs_isfinal[-1]
        proj_queries, proj_tokens = None, None

        if self.aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_logits": a,
                    "pred_boxes": b,
                }
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
            ]
            if outputs_isfinal is not None:
                assert len(outputs_isfinal[:-1]) == len(out["aux_outputs"])
                for i in range(len(outputs_isfinal[:-1])):
                    out["aux_outputs"][i]["pred_isfinal"] = outputs_isfinal[i]
        return out
        # return hs

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_boxes': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, eos_coef, losses, temperature):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.losses = losses
        self.temperature = temperature
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_isfinal(self, outputs, targets, positive_map, indices, num_boxes):
        """This loss is used in some referring expression dataset (specifically Clevr-REF+)
        It trains the model to predict which boxes are being referred to (ie are "final")
        Eg if the caption is "the cube next to the cylinder", MDETR will detect both the cube and the cylinder.
        However, the cylinder is an intermediate reasoning step, only the cube is being referred here.
        """
        idx = self._get_src_permutation_idx(indices)
        src_isfinal = outputs["pred_isfinal"][idx].squeeze(-1)
        target_isfinal = torch.cat([t["isfinal"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_isfinal = F.binary_cross_entropy_with_logits(src_isfinal, target_isfinal, reduction="none")

        losses = {}
        losses["loss_isfinal"] = loss_isfinal.sum() / num_boxes
        acc = (src_isfinal.sigmoid() > 0.5) == (target_isfinal > 0.5)
        if acc.numel() == 0:
            acc = acc.sum()
        else:
            acc = acc.float().mean()
        losses["accuracy_isfinal"] = acc

        return losses

    # def loss_labels(self, outputs, targets, positive_map, indices, num_boxes):
    #     """Classification loss (NLL)
    #     targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    #     """
    #
    #     logits = outputs["pred_logits"].log_softmax(-1)  # BS x (num_queries) x (num_tokens)
    #
    #     src_idx = self._get_src_permutation_idx(indices)
    #     tgt_idx = []
    #     offset = 0
    #     for i, (_, tgt) in enumerate(indices):
    #         tgt_idx.append(tgt + offset)
    #         offset += len(targets[i]["boxes"])
    #     tgt_idx = torch.cat(tgt_idx)
    #
    #     ## 修改
    #     positive_map = []
    #     for i in range(len(targets)):
    #         print("targets[i][labels]: ")
    #         print(targets[i]["labels"])
    #         if targets[i]["labels"].cpu().item() == 0 :
    #             positive_map.append([0, 1])
    #         else:
    #             positive_map.append([1, 0])
    #     positive_map = torch.from_numpy(np.array(positive_map))
    #
    #     tgt_pos = positive_map
    #     target_sim = torch.zeros_like(logits)
    #     target_sim[:, :, -1] = 1
    #     target_sim[src_idx] = tgt_pos
    #
    #     loss_ce = -(logits * target_sim).sum(-1)
    #
    #     eos_coef = torch.full(loss_ce.shape, self.eos_coef, device=target_sim.device)
    #     one_tensor = torch.tensor([1.,1.,1.,1.]).to(eos_coef.device)
    #     eos_coef[src_idx] = one_tensor
    #
    #     loss_ce = loss_ce * eos_coef
    #     loss_ce = loss_ce.sum() / num_boxes
    #
    #     losses = {"loss_ce": loss_ce}
    #
    #     return losses

    def loss_labels(self, outputs, targets, positive_map, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # print("cross_entropy: ")
        # print(src_logits.transpose(1, 2).shape)
        # print(target_classes.shape)
        # print(self.empty_weight.shape)
        # exit()

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        # if log:
        #     # TODO this should probably be a separate loss, not hacked in this one here
        #     losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_contrastive_align(self, outputs, targets, positive_map, indices, num_boxes):
        bs = outputs["proj_queries"].shape[0]
        tokenized = outputs["tokenized"]

        normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        logits = (
            torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.temperature
        )  # BS x (num_queries) x (num_tokens)

        # construct a map such that positive_map[k, i,j] = True iff query i is associated to token j in batch item k
        # For efficency, the construction happens on CPU, then the whole matrix is transferred to GPU in one go.
        positive_map = torch.zeros(logits.shape, dtype=torch.bool)
        for i, ((idx_src, idx_tgt), tgt) in enumerate(zip(indices, targets)):
            if "tokens_positive" in tgt:
                cur_tokens = [tgt["tokens_positive"][j] for j in idx_tgt]
            else:
                cur_tokens = [tgt["tokens"][j] for j in idx_tgt]

            for j, tok_list in enumerate(cur_tokens):
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(i, beg)
                    end_pos = tokenized.char_to_token(i, end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    positive_map[i, idx_src[j], beg_pos : end_pos + 1].fill_(True)

        positive_map = positive_map.to(logits.device)
        positive_logits = -logits.masked_fill(~positive_map, 0)
        negative_logits = logits  # .masked_fill(positive_map, -1000000)

        boxes_with_pos = positive_map.any(2)
        pos_term = positive_logits.sum(2)
        neg_term = negative_logits.logsumexp(2)

        nb_pos = positive_map.sum(2) + 1e-6

        box_to_token_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~boxes_with_pos, 0).sum()

        tokens_with_pos = positive_map.any(1)
        pos_term = positive_logits.sum(1)
        neg_term = negative_logits.logsumexp(1)

        nb_pos = positive_map.sum(1) + 1e-6

        tokens_to_boxes_loss = ((pos_term / nb_pos + neg_term)).masked_fill(~tokens_with_pos, 0).sum()
        tot_loss = (box_to_token_loss + tokens_to_boxes_loss) / 2

        return {"loss_contrastive_align": tot_loss / num_boxes}

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        ## Count the number of predictions that are NOT "no-object" (which is the last class)
        # normalized_text_emb = outputs["proj_tokens"]  # BS x (num_tokens) x hdim
        # normalized_img_emb = outputs["proj_queries"]  # BS x (num_queries) x hdim

        # logits = torch.matmul(
        #    normalized_img_emb, normalized_text_emb.transpose(-1, -2)
        # )  # BS x (num_queries) x (num_tokens)
        # card_pred = (logits[:, :, 0] > 0.5).sum(1)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # print("src_boxes: ")
        # print(src_boxes)
        # print(target_boxes)
        # exit()

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), box_ops.box_cxcywh_to_xyxy(target_boxes))
        )

        # print("target: ")
        # print(target_boxes)
        # print(src_boxes)
        # exit()

        # loss_giou = 1 - torch.diag(
        #     box_ops.generalized_box_iou(box_ops.box_cxcywh_to_xyxy(src_boxes), target_boxes)
        # )
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, positive_map, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = NestedTensor.from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, positive_map, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "isfinal": self.loss_isfinal,
            "contrastive_align": self.loss_contrastive_align,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, positive_map, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, positive_map):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if dist.is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / dist.get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, positive_map, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, positive_map, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class ContrastiveCriterion(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, pooled_text, pooled_image):

        normalized_text_emb = F.normalize(pooled_text, p=2, dim=1)
        normalized_img_emb = F.normalize(pooled_image, p=2, dim=1)

        logits = torch.mm(normalized_img_emb, normalized_text_emb.t()) / self.temperature
        labels = torch.arange(logits.size(0)).to(pooled_image.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        loss = (loss_i + loss_t) / 2.0
        return loss

def build(args):
    # if args.dataset_file == "ytvos":
    #     num_classes = 40
    num_classes = 1
    device = torch.device(args.device)

    backbone = build_backbone(args)

    # model = Model(backbone)
    #
    # return model, None, None, None

    transformer = build_transformer(args)

    model = Model(
        backbone,
        transformer,
        num_classes=num_classes,
        num_frames=args.num_frames,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        contrastive_hdim=args.contrastive_loss_hdim,
        contrastive_loss=args.contrastive_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        predict_final=args.predict_final,
    )



    if args.mask_model != "none":
        model = Modelsegm(
            model,
            mask_head=args.mask_model,
            freeze_detr=(args.frozen_weights is not None),
        )
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.ce_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    if args.contrastive_loss:
        weight_dict["contrastive_loss"] = args.contrastive_loss_coef
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.predict_final:
        weight_dict["loss_isfinal"] = 1

    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    if args.predict_final:
        losses += ["isfinal"]
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]

    criterion = None
    if not args.no_detection:
        criterion = SetCriterion(
            num_classes,
            matcher=matcher,
            eos_coef=args.eos_coef,
            losses=losses,
            temperature=args.temperature_NCE,
        )
        criterion.to(device)

    if args.contrastive_loss:
        contrastive_criterion = ContrastiveCriterion(temperature=args.temperature_NCE)
        contrastive_criterion.to(device)
    else:
        contrastive_criterion = None

    # postprocessors = {'bbox': PostProcess()}
    # if args.masks:
    #     postprocessors['segm'] = PostProcessSegm()
    return model, criterion, contrastive_criterion, weight_dict
