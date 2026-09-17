import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from layer_replay import replay_logps


def original(model, rec):
    past = None
    values = []
    p, c = rec['input_ids'], rec['output_ids']
    for i, target in enumerate(c):
        feed = p if i == 0 else [c[i-1]]
        out = model.model(input_ids=torch.tensor([feed]),
                          attention_mask=torch.ones((1, len(p)+i), dtype=torch.long),
                          past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = model.lm_head(out.last_hidden_state[:, -1, :]).float()[0] / .8
        values.append(logits[target] - torch.logsumexp(logits, dim=0))
    return torch.stack(values)


def main():
    torch.set_num_threads(2)
    torch.manual_seed(31)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=8, attention_dropout=0.)
    config._attn_implementation = 'sdpa'
    model = Qwen3ForCausalLM(config).eval()
    for prompt, completion in ((7, 5), (17, 13)):
        rec = dict(input_ids=list(range(1, prompt+1)),
                   output_ids=list(range(3, completion+3)),
                   sampling=dict(temperature=.8, grammar=None))
        weights = torch.linspace(-.7, 1.1, completion)
        model.zero_grad(set_to_none=True)
        ref = original(model, rec)
        (ref*weights).sum().backward()
        grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        for use_cp in (False, True):
            model.zero_grad(set_to_none=True)
            actual = replay_logps(model, rec, use_cp)
            torch.testing.assert_close(actual, ref, atol=2e-5, rtol=2e-5)
            (actual*weights).sum().backward()
            for n,p in model.named_parameters():
                if n in grads:
                    assert p.grad is not None, n
                    torch.testing.assert_close(p.grad, grads[n], atol=2e-5, rtol=2e-4, msg=n)
            print('TINY_FORWARD_AND_GRADIENT=passed', prompt, completion, use_cp, flush=True)
    print('SCOPE=tiny_FP32_only; real_quantized_LoRA_NOT_TESTED')


if __name__ == '__main__':
    main()
