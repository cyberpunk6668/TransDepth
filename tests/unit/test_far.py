import torch

from transdepth.data.schema import PatchTargets
from transdepth.far.relation_loss import relation_kl_loss
from transdepth.far.teacher import teacher_rows
from transdepth.models.types import AttentionTrace


def test_teacher_preserves_rows_mass_and_complement() -> None:
    logits = torch.tensor([[0.1, 0.4, -0.2, 0.8], [0.2, -0.1, 0.3, 0.5]])
    key_ids = torch.tensor([1, 3])
    teacher = teacher_rows(
        logits,
        key_ids,
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, False]),
        beta=0.5,
    )
    base = logits.softmax(-1)
    complement = torch.tensor([0, 2])
    assert torch.allclose(teacher.sum(-1), torch.ones(2), atol=1e-7)
    assert torch.equal(teacher[:, complement], base[:, complement])
    assert torch.allclose(teacher[:, key_ids].sum(-1), base[:, key_ids].sum(-1), atol=1e-7)
    assert torch.equal(teacher[1], base[1])


def test_beta_zero_is_exact_identity() -> None:
    logits = torch.randn(3, 7)
    result = teacher_rows(
        logits,
        torch.tensor([1, 5]),
        torch.ones(3),
        torch.ones(2),
        torch.ones(3, dtype=torch.bool),
        beta=0.0,
    )
    assert torch.equal(result, logits.float().softmax(-1))


def _patch_targets(states: torch.Tensor, depths: torch.Tensor) -> PatchTargets:
    zeros = torch.zeros_like(depths, dtype=torch.float32)
    return PatchTargets(states.to(torch.int8), depths.float(), zeros, zeros, zeros)


def test_relation_loss_has_student_gradient_and_detached_teacher() -> None:
    torch.manual_seed(4)
    base_q = torch.randn(1, 1, 7, 4)
    base_k = torch.randn(1, 1, 7, 4)
    student_q = base_q.clone().requires_grad_()
    student_k = base_k.clone().requires_grad_()
    trace = AttentionTrace(student_q, student_k, base_q, base_k, 15, (2,), prefix_tokens=5)
    target = _patch_targets(torch.tensor([[1, 1]]), torch.tensor([[1.0, 2.0]]))
    result = relation_kl_loss(trace, [target], beta=0.5)
    assert result.loss.item() >= 0
    result.loss.backward()
    assert student_q.grad is not None and student_q.grad.abs().sum() > 0
    assert student_k.grad is not None and student_k.grad.abs().sum() > 0
    assert base_q.grad is None and base_k.grad is None


def test_empty_relation_returns_connected_zero() -> None:
    q = torch.randn(1, 1, 6, 4, requires_grad=True)
    k = torch.randn(1, 1, 6, 4, requires_grad=True)
    trace = AttentionTrace(q, k, q.detach(), k.detach(), 15, (0,), prefix_tokens=5)
    target = _patch_targets(torch.tensor([[-1]]), torch.tensor([[float("nan")]]))
    result = relation_kl_loss(trace, [target], beta=0.5)
    assert result.loss.item() == 0.0
    result.loss.backward()
    assert q.grad is not None and torch.count_nonzero(q.grad) == 0
