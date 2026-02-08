# coding: utf-8
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np

def tile(x: Tensor, count: int, dim=0) -> Tensor:

    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def beam_search(
    decoder,
    size: int,
    bos_index: int,
    eos_index: int,
    pad_index: int,
    encoder_output: Tensor,
    src_mask: Tensor,
    max_output_length: int,
    alpha: float,
    n_best: int = 1,
) -> (np.array, np.array):

    assert size > 0, "Beam size must be >0."
    assert n_best <= size, "Can only return {} best hypotheses.".format(size)


    batch_size = src_mask.size(0)

    encoder_output = tile(
        encoder_output.contiguous(), size, dim=0
    ) 
    src_mask = tile(src_mask, size, dim=0) 

    trg_mask = src_mask.new_ones([1, 1, 1]) 

    
    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device
    ) 

    
    beam_offset = torch.arange(
        0, batch_size * size, step=size, dtype=torch.long, device=encoder_output.device
    ) 

   
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device,
    )

   
    topk_log_probs = torch.zeros(batch_size, size, device=encoder_output.device)
    topk_log_probs[:, 1:] = float("-inf")

    
    hypotheses = [[] for _ in range(batch_size)]

    results = {
        "predictions": [[] for _ in range(batch_size)],
        "scores": [[] for _ in range(batch_size)],
        "attentions" : [[] for _ in range(batch_size)],
       
    }

    for step in range(max_output_length):

        decoder_input = alive_seq  

        from dataloader import subsequent_mask
        logits = decoder.decode(encoder_output, 
                            src_mask,
                            decoder_input,
                            subsequent_mask(decoder_input.size(1)).to(encoder_output.device).long(),
                            )
        logits = logits.reshape( (decoder_input.size(0), -1, logits.size(-1)))

        logits = logits[:, -1]

        
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs.clone()

        output_size = log_probs.size(-1)

        
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

       
        curr_scores = curr_scores.reshape(-1, size * output_size)

       
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            
            topk_log_probs = topk_scores * length_penalty
        else:
            topk_log_probs = topk_scores.clone()

       
        try:
            topk_beam_index = topk_ids.div(output_size, rounding_mode='floor')
        except:
           
            topk_beam_index = topk_ids.div(output_size) 
        topk_ids = topk_ids.fmod(output_size) 

     
        batch_index = topk_beam_index + beam_offset[
            : topk_beam_index.size(0)
        ].unsqueeze(1)
        
        select_indices = batch_index.view(-1)

     
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), topk_ids.view(-1, 1)], -1
        )

       

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(True)
       
        end_condition = is_finished[:, 0].eq(True)

       
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
           
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = torch.nonzero(is_finished[i], as_tuple=False).view(-1)
               
                for j in finished_hyp:
                    
                    if torch.nonzero(predictions[i, j, 1:] == eos_index, as_tuple=False).numel() < 2:
                      
                        hypotheses[b].append((topk_scores[i, j], predictions[i, j, 1:]))

              
                if end_condition[i]:
                    best_hyp = sorted(hypotheses[b], key=lambda x: x[0], reverse=True)
                   
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
                        
            non_finished = torch.nonzero(end_condition.eq(False), as_tuple=False).view(-1)
         
            if len(non_finished) == 0:
                break
           
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished).view(
                -1, alive_seq.size(-1)
            )

        
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

    final_outputs = [ [ rr.detach().cpu() for rr in br ] for br in results["predictions"] ]
    final_scores = [ [ rr.detach().cpu().item() for rr in br ] for br in results["scores"] ]
    


    return final_outputs, final_scores
