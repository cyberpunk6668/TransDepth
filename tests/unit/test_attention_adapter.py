import torch
import torch.nn.functional as F
from torch import nn

from transdepth.models.backbone.attention_adapter import FARSelfAttention


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 2
        self.scale = 0.5
        self.qkv = nn.Linear(8, 24, bias=True)
        self.proj = nn.Linear(8, 8, bias=True)
        self.proj_drop = nn.Identity()

    def forward(self, inputs, attn_bias=None, rope=None):
        assert attn_bias is None and rope is None
        batch, tokens, width = inputs.shape
        packed = self.qkv(inputs).reshape(batch, tokens, 3, 2, 4)
        q, k, v = (item.transpose(1, 2) for item in torch.unbind(packed, dim=2))
        output = F.scaled_dot_product_attention(q, k, v)
        return self.proj(output.transpose(1, 2).reshape(batch, tokens, width))


def test_adapter_preserves_tensor_and_single_item_list_paths() -> None:
    torch.manual_seed(8)
    original = FakeAttention()
    inputs = torch.randn(2, 7, 8)
    expected = original(inputs)
    adapter = FARSelfAttention(
        original, selected_block=3, selected_head=1, rank=2, seed=17, prefix_tokens=5
    )
    assert torch.equal(adapter(inputs), expected)
    listed = adapter.forward_list([inputs], rope_list=[None])
    assert len(listed) == 1 and torch.equal(listed[0], expected)
    adapter.set_capture_trace(True)
    adapter(inputs)
    trace = adapter.consume_trace()
    assert trace.q_student.shape == (2, 1, 7, 4)
    assert trace.selected_heads == (1,)
