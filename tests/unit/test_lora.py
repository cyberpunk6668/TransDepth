import torch
from torch import nn

from transdepth.models.backbone.selected_qk_lora import SelectedQKLoRA


def test_zero_initialized_lora_is_identity_and_parameter_count_is_exact() -> None:
    base = nn.Linear(8, 24, bias=True)
    adapter = SelectedQKLoRA(base, num_heads=2, selected_head=1, rank=2, seed=3)
    inputs = torch.randn(2, 5, 8)
    student, original = adapter.forward_with_base(inputs)
    assert torch.equal(student, original)
    trainable = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    assert trainable == 2 * 2 * (8 + 4)


def test_only_selected_qk_rows_change_and_merge_is_equivalent() -> None:
    torch.manual_seed(2)
    base = nn.Linear(8, 24, bias=True)
    adapter = SelectedQKLoRA(
        base, num_heads=2, selected_head=1, rank=2, eta=1.0, seed=3
    )
    with torch.no_grad():
        adapter.q_up.weight.fill_(0.2)
        adapter.k_up.weight.fill_(-0.1)
    inputs = torch.randn(2, 5, 8)
    student, original = adapter.forward_with_base(inputs)
    changed = (student - original).abs().sum(dim=(0, 1)) > 0
    expected = torch.zeros(24, dtype=torch.bool)
    expected[4:8] = True
    expected[12:16] = True
    assert torch.equal(changed, expected)
    adapter.merge_into_base_()
    assert torch.allclose(adapter.base(inputs), student, atol=1e-6, rtol=1e-6)
